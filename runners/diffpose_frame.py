import os
import logging
import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn


from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps, limb_cond
from common.data_utils import fetch_me, read_3d_data_me, create_2d_data
from common.generators import PoseGenerator_gmm
from common.loss import mpjpe, p_mpjpe


class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda()
        self.src_mask = self.src_mask[:,:,:17]
        self.train_loader = None
        self.valid_loader = None
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.best_p1, self.best_epoch = 1000, 0

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            self.dataset = read_3d_data_me(dataset)
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
                print('==> Selected actions: {}'.format(self.action_filter))
        else:
            raise KeyError('Invalid dataset')
        

    def print_param(self):
        if self.model_diff != None:
            overall_param = sum(param.numel() for param in self.model_diff.parameters())
            global_param = sum(param.numel() for param in self.model_diff.module.gconv_input1.parameters())
            local_param = sum(param.numel() for param in self.model_diff.module.local_emb.parameters())
            local_param += sum(param.numel() for param in self.model_diff.module.local_attention.parameters())            
            temp_param = sum(param.numel() for param in self.model_diff.module.temperal_conv.parameters())
            temp_param += sum(param.numel() for param in self.model_diff.module.gconv_input_t.parameters())
            limb_param = sum(param.numel() for param in self.model_diff.module.limb_linear.parameters())
            limb_param += sum(param.numel() for param in self.model_diff.module.limb_emb_layer.parameters())
            print(f"overall: {overall_param};, global: {global_param};, local: {local_param};, temp: {temp_param};, limb: {limb_param};,")
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(self.model_diff, (torch.rand(2, 17, 21).cuda(), torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda(), torch.randint(low=0, high=self.num_timesteps,
                                  size=(2 // 2 + 1,)).cuda(), torch.rand(2, 17, 64).cuda(),  torch.rand(2, 4, 5000, 6).cuda(), None))
            print(flops.by_module())

          


    # create diffusion model
    def create_diffusion_model(self, model_path = None, model_limb_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)

        # edges = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 4],
        #          [5, 7], [6, 8],  [7, 9], [8, 10], [0, 6], [0, 5],
        #           [9,11], [10, 11]], dtype=torch.long)
        # edges = torch.tensor([[0, 1], [0, 2], [1, 3],
        #                     [2, 4], [5, 6], [5, 7],
        #                     [0, 7], [0, 6], [6, 8], [8,10],
        #                     [7, 9], [9, 11]], dtype=torch.long)
        # edges = torch.tensor([[0, 1], [0, 2], [1, 3],
        #                     [2, 4], [11, 10], [11, 9],
        #                     [0, 9], [0, 10], [10, 8], [8,6],
        #                     [7, 9], [7, 5], [0, 11]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)

        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model
        if model_limb_path:
            checkpoint = torch.load(model_limb_path)
            model_ref = GCNdiff(adj.cuda(), config).cuda()
            model_ref = torch.nn.DataParallel(model_ref)


            model_ref.load_state_dict(checkpoint[0])
        
            self.model_diff.module.limb_linear.load_state_dict(model_ref.module.limb_linear.state_dict())
            print("limb len loaded..")
            del model_ref



        if model_path:
            # # add on
            # # model_diff_ref = GCNdiff_ref(adj.cuda(), config).cuda()
            # module_names = ["temb", "atten_layers", "gconv_layers"]
            # states = torch.load(model_path)
            # for key, value in states[0].items():
            #     for key_selected in module_names:
            #         if key_selected in key:
            #             print(key)
            #             self.model_diff.state_dict()[key].copy_(value)
           



            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
        
        
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        # edges = torch.tensor([[0, 1], [0, 2], [1, 3],
        #                     [2, 4], [11, 10], [11, 9],
        #                     [0, 9], [0, 10], [10, 8], [8,6],
        #                     [7, 9], [7, 5], [0, 11]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_pose = GCNpose(adj.cuda(), config).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states[0])
        else:
            logging.info('initialize model randomly')

    def train(self):
        cudnn.benchmark = True

        # debug
        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
            
            if self.valid_loader == None:
                data_loader = self.valid_loader = data.DataLoader(
                    PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid, test=True),
                    batch_size=config.training.batch_size, shuffle=False, 
                    num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset') 



        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        self.best_p1, self.best_epoch = 1000, 0
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
            if self.train_loader == None:
                data_loader = self.train_loader = data.DataLoader(
                    PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                    batch_size=config.training.batch_size, shuffle=True,\
                        num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset')
        
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()

            for i, (targets_uvxyz, targets_noise_scale, input_feat, targets_3d_pred, _, _, radar, limb_len_gt) in enumerate(self.train_loader):
                data_time += time.time() - data_start
                step += 1

                # to cuda
                targets_uvxyz, targets_noise_scale, targets_3d_pred, limb_len_gt = \
                    targets_uvxyz.to(self.device), targets_noise_scale.to(self.device), targets_3d_pred.to(self.device), limb_len_gt.to(self.device)
                # targets_uvxyz: ([1024, 17, 5]), targets_noise_scale: ([1024, 17, 5]); targets_3d: ([1024, 17, 3])
                # print("mean: ", targets_uvxyz[0,:5,:]-targets_uvxyz[0,0,:])
                # print("var: ", targets_noise_scale[0,:5,:])
                # generate nosiy sample based on seleted time t and beta
                n = targets_3d_pred.size(0)
                # print(n)
                x = targets_uvxyz

                e = torch.randn_like(x)
                b = self.betas        
                # print(b.size())    
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                # print(t.size())    
                e = e*(targets_noise_scale)
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # print(t[:8])
                # print((1-b).cumprod(dim=0).index_select(0, t)[:8])    

                # generate x_t (refer to DDIM equation)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                # predict noise
                output_noise, limb_len_pred = self.model_diff(x, src_mask, t.float(), input_feat, radar, limb_len_gt)
                # e = e + 100*limb_cond(x, e, a, limb_len_gt)
                loss_diff = (e[:, :, -3:] - output_noise[:, :, -3:]).square().sum(dim=(1, 2)).mean(dim=0)
                

                limb_loss =  (limb_len_pred - limb_len_gt).abs().sum(dim=-1).mean(dim=0)
                loss_diff = loss_diff + limb_loss*10
                # if False:
                #     for param in self.model_diff.module.parameters():
                #         param.requires_grad = False
                #     for param in self.model_diff.module.limb_linear.parameters():
                #         param.requires_grad = True
                #     output_noise, limb_len_pred = self.model_diff(x, src_mask, t.float(), input_feat, radar, limb_len_gt)
                #     loss_diff = (limb_len_pred - limb_len_gt).abs().sum(dim=-1).mean(dim=0)
                # else:
                #     for param in self.model_diff.module.parameters():
                #         param.requires_grad = True
                #     for param in self.model_diff.module.limb_linear.parameters():
                #         param.requires_grad = False
                #     output_noise, limb_len_pred = self.model_diff(x, src_mask, t.float(), input_feat, radar, limb_len_gt)
                #     loss_diff = (e[:, :, -3:] - output_noise[:, :, -3:]).square().sum(dim=(1, 2)).mean(dim=0)
                #     limb_loss =  (limb_len_pred - limb_len_gt).abs().sum(dim=-1).mean(dim=0)
                #     loss_diff = loss_diff #+ limb_loss*10
                
                optimizer.zero_grad()
                loss_diff.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if i%10 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(self.train_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                
            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            
                logging.info('test the performance of current model')

                p1, p2 = self.test_hyber(is_train=True)

                if p1 < self.best_p1:
                    self.best_p1 = p1
                    self.best_epoch = epoch
                logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(self.best_epoch, self.best_p1, epoch, p1, p2))
    

    


    def test_hyber(self, is_train=False):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
            
            if self.valid_loader == None:
                data_loader = self.valid_loader = data.DataLoader(
                    PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid, test=True),
                    batch_size=config.training.batch_size, shuffle=False, 
                    num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset') 

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        # self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',\
        #     'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        self.test_action_list = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]
        action_error_sum = define_error_list(self.test_action_list) 

        vis_idx = 0  
        target_idx = 24543    
        limb_loss = None 

        prediction_list = []
        for i, (targets_3d_pred, input_noise_scale, input_feat, targets_3d, input_action, _, radar, limb_len_gt) in enumerate(self.valid_loader):
            data_time += time.time() - data_start

            targets_3d, input_noise_scale, input_feat, targets_3d_pred, limb_len_gt = \
                targets_3d.to(self.device), input_noise_scale.to(self.device), input_feat.to(self.device), targets_3d_pred.to(self.device), limb_len_gt.to(self.device)

            # build uvxyz
            # inputs_xyz = self.model_pose(input_2d, src_mask)
            # targets_3d_pred = torch.cat([targets_3d_pred.clone(), targets_3d_pred.clone()],dim=2)      
            targets_3d_pred[:, :, :] -= targets_3d_pred[:, :1, :] 
            # input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
                        
            # generate distribution
            targets_3d_pred = targets_3d_pred.repeat(test_times,1,1)
            input_noise_scale = input_noise_scale.repeat(test_times,1,1)
            input_feat = input_feat.repeat(test_times,1,1)
            radar = radar.repeat(test_times,1,1,1)
            limb_len_gt = limb_len_gt.repeat(test_times,1)
            

            # select diffusion step
            t = torch.ones(targets_3d_pred.size(0)).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps
            
            # prepare the diffusion parameters
            x = targets_3d_pred.clone()
            e = torch.randn_like(targets_3d_pred)
            b = self.betas   
            e = e*input_noise_scale * 1
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            # x = x * a.sqrt() + e * (1.0 - a).sqrt()
            
            output_uvxyz, _, limb_len_pred = generalized_steps(x, input_feat, radar, limb_len_gt, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            output_uvxyz = output_uvxyz[-1]            
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,output_uvxyz.shape[-1]),0)
            output_xyz = output_uvxyz[:,:,-3:]
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]
            
            epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))
            if limb_loss == None:
                limb_loss = (limb_len_pred - limb_len_gt).abs().sum(dim=-1)
            else:
                limb_loss = torch.cat([limb_loss, (limb_len_pred - limb_len_gt).abs().sum(dim=-1)], dim=0)
            

            prediction_list.append(output_uvxyz[:, :, -3:].cpu().detach().numpy())

            # flag += 100
            # save_prediction_3d(x[0, :, -3:], targets_3d[0], epochs=f"original_{flag}", mpj = torch.pow(x[0, :, -3:] - targets_3d[0], 2).mean())
            # save_prediction_3d(output_uvxyz[0, :, -3:], targets_3d[0], epochs=f"optimized_{flag}", mpj = torch.pow(output_uvxyz[0, :, -3:] - targets_3d[0], 2).mean())
            
            data_start = time.time()
            
            action_error_sum = test_calculation(output_xyz, targets_3d, input_action, action_error_sum, None, None)
            
            if i%1 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(self.valid_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f} | Limb-loss: {e3: .4f}'\
                .format(batch=i + 1, size=len(self.valid_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                    e2=epoch_loss_3d_pos_procrustes.avg, e3=limb_loss.mean(dim=0)))
        
        p1, p2 = print_error(None, action_error_sum, is_train)


        return p1, p2