import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import optim

import numpy as np
import scipy.sparse as sp
import copy, math
import logging
from tqdm import tqdm
import time, os


from .ChebConv import ChebConv, _GraphConv, _ResChebGC
from .GraFormer import *
from ..p4trans import P4Transformer_feat
from ..pointTrans import PointTransformerReg_feat

from .utils import *
from .ema import EMAHelper
from .pretrain_save import phase1_save
from .generators import PoseGenerator_gmm
from .loss import *




### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    
class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ### time embedding ###
        self.temb_proj = torch.nn.Linear(emd_dim,hid_dim)

    def forward(self, x, temb, condition = None):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        if condition != None:
            out = out + condition
        out = self.gconv2(out, self.adj)
        return residual + out

class GCNdiff(nn.Module):
    def __init__(self, adj, config, global_feat_size = 64, past_frames = 6, radar_input_c = 6, 
                 global_flag = 1, local_flag = 1, temp_flag = 1, limb_flag = 1):
        super(GCNdiff, self).__init__()

        ### Configurations ###
        self.adj = adj
        self.config = config
        ### Load gcn configuration ###
        con_gcn = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, n_pts = \
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim, \
                con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
                
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim*4

        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda()
        self.src_mask = self.src_mask[:,:,:17]

        ### Generate Graphformer  ###
        self.n_layers = num_layers
        self.feature_size = global_feat_size
        self.past_frames = past_frames

        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_input2 = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
        _gconv_input3 = ChebConv(in_c=3, out_c=self.hid_dim, K=2)


        
        _gconv_layers = []
        _attention_layer = []

        dim_model = self.hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        self.n_pts = 17
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=self.n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        
        self.gconv_input2 = _gconv_input2
        self.gconv_input3 = _gconv_input3
        
        
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3 * (self.past_frames+1), K=2)
        
        
        ### diffusion configuration  ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim,self.emd_dim),
            torch.nn.Linear(self.emd_dim,self.emd_dim),
        ])

        
        self.global_flag = global_flag
        self.local_flag = local_flag
        self.temp_flag = temp_flag
        self.limb_flag = limb_flag

        #############################
        ### Global embedding start###
        #############################

        if self.global_flag == 1:
            ### global gcn projector ###
            _gconv_input1 = ChebConv(in_c=self.feature_size, out_c=self.hid_dim, K=2)
            self.gconv_input1 = _gconv_input1

        ############################
        ### Local embedding start###
        ############################

        ### local mlp ###
        if self.local_flag == 1:
            
            self.input_dim = radar_input_c
            self.local_dim = self.hid_dim
            self.local_emb = nn.Module()   
            self.local_emb.dense = nn.ModuleList([
                torch.nn.Linear(self.input_dim,self.local_dim // 2),
                torch.nn.Linear(self.local_dim // 2,self.local_dim),
            ])
            
            ### local attn ###
            self.local_attn_iter = 5
            local_attn = MultiHeadedAttention(n_head, self.local_dim)
            _local_attention = []
            for i in range(self.local_attn_iter):
                _local_attention.append(c(local_attn))
            self.local_attention = nn.ModuleList(_local_attention)
        if self.local_flag == 2:
            # To be overwrited
            self.input_dim = 6
            self.local_dim = self.hid_dim
            self.local_emb = nn.Module()   
            self.local_emb.dense = nn.ModuleList([
                torch.nn.Linear(self.input_dim,self.local_dim // 2),
                torch.nn.Linear(self.local_dim // 2,self.local_dim),
            ])
            
            ### local attn ###
            self.local_attn_iter = 5
            local_attn = MultiHeadedAttention(n_head, self.local_dim)
            _local_attention = []
            for i in range(self.local_attn_iter):
                _local_attention.append(c(local_attn))
            self.local_attention = nn.ModuleList(_local_attention)

        ###############################
        ### temporal embedding start###
        ###############################

        if self.temp_flag == 1:
            ### shared gcn encoder ###
            _gconv_input_t = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
            self.gconv_input_t = _gconv_input_t

            ### temperal convolutional encoder ###
            self.temperal_hid_dim = self.hid_dim * 17
            self.temperal_conv = nn.Sequential(
                nn.Conv1d(self.temperal_hid_dim, self.temperal_hid_dim, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv1d(self.temperal_hid_dim, self.temperal_hid_dim, kernel_size = 3, padding = 1),
                nn.MaxPool1d(kernel_size = self.past_frames)
            )
            
        ###########################
        ### limb embedding start###
        ###########################
        if self.limb_flag == 1:
            # limb mlp decoder
            self.limb_size = 16
            limb_feature_size = 1024
            self.limb_linear = nn.Sequential(
                torch.nn.LayerNorm(17 * self.feature_size),
                torch.nn.Linear(17 * self.feature_size,limb_feature_size),
                torch.nn.ReLU(),
                # torch.nn.Dropout(p=0.5),
                torch.nn.Linear(limb_feature_size, self.limb_size),
            )

            # limb token conditional projector
            self.limb_emb_layer = nn.Module()
            
            self.limb_emb_layer.dense = nn.ModuleList([
                torch.nn.Linear(self.limb_size,self.emd_dim),
                torch.nn.Linear(self.emd_dim,self.emd_dim),
            ])

        if self.limb_flag == 2:
            # limb mlp decoder
            self.limb_size = 16
            self.limb_linear = nn.Sequential(
                torch.nn.LayerNorm((self.past_frames+1)*self.n_pts * 3),
                torch.nn.Linear((self.past_frames+1)*self.n_pts * 3,1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, self.limb_size),
            )

            # limb token conditional projector
            self.limb_emb_layer = nn.Module()
            
            self.limb_emb_layer.dense = nn.ModuleList([
                torch.nn.Linear(self.limb_size,self.emd_dim),
                torch.nn.Linear(self.emd_dim,self.emd_dim),
            ])




    def forward(self, x, t, cemd, radar, limb_len):

        # timestep embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        '''
        ##########################
            Global Embedding
        ##########################
        '''
        if self.global_flag == 1:
            global_emb = self.gconv_input1(cemd, self.adj)


        '''
        ##########################
            Local Embedding
        ##########################
        '''
        if self.local_flag == 1:
            thre = 0.04
            local_emb_list = []

            b, t, n, c = radar.shape
            point_cloud = radar.view(b, t * n, c)

            for joint_id in range(17):
                # select local 50 points, with 0 paddings
                anchor = x[:, [joint_id], -3:]
                dist = torch.sum(torch.pow(point_cloud[:, :, :3] - anchor, 2), dim=-1)
                index_sort = torch.argsort(dist, dim = -1).unsqueeze(-1).expand(-1,-1,c)
                point_cloud_select = torch.gather(point_cloud, 1, index_sort)[:,:50,:]

                # scaling by the number of points
                dist_sort = torch.sum(torch.pow(point_cloud_select[:,:,:3] - anchor, 2), dim=-1)
                dist_scale = torch.sum(dist_sort.lt(thre), dim=-1, keepdim = True) / 50


                # shared embedding layers
                local_emb = self.local_emb.dense[0](point_cloud_select)
                local_emb = nonlinearity(local_emb)
                local_emb = self.local_emb.dense[1](local_emb)
                

                # local self attention
                for i in range(self.local_attn_iter):
                    local_emb = local_emb + self.local_attention[i](local_emb, local_emb, local_emb)

                # max pooling
                local_emb = torch.mean(local_emb, dim = 1)
                local_score = dist_scale
                local_emb = local_emb * local_score
                local_emb_list.append(local_emb)

            # concatenation
            local_emb = torch.stack(local_emb_list, dim=1)
          
        
        '''
        ##########################
            Temperal Embedding
        ##########################
        '''
        if self.temp_flag == 1:
            x_past_list = []
            for i in range(self.past_frames):
                x_curr = x[:, :, i*3:(i+1)*3]
                x_curr = self.gconv_input_t(x_curr, self.adj)   
                x_past_list.append(x_curr)   

            x_past = torch.stack(x_past_list, dim=1)
            past_emb = x_past.view(-1, x_past.size(1), x_past.size(2) * x_past.size(3)).permute(0,2,1)
            past_emb = self.temperal_conv(past_emb).squeeze().view(-1, x_past.size(2), x_past.size(3))



        '''
        ##########################
            Limb Embedding
        ##########################
        '''
        if self.limb_flag == 1:
            limb_len_pred = self.limb_linear(cemd.view(-1, 17 * self.feature_size))

            if limb_len == None:
                limb_len = limb_len_pred

            limb_emb = self.limb_emb_layer.dense[0](limb_len)
            limb_emb = nonlinearity(limb_emb)
            limb_emb = self.limb_emb_layer.dense[1](limb_emb)

        if self.limb_flag == 2:
            limb_len_pred = self.limb_linear(x.reshape(-1, (self.past_frames+1)*self.n_pts * 3))

            if limb_len == None:
                limb_len = limb_len_pred

            limb_emb = self.limb_emb_layer.dense[0](limb_len)
            limb_emb = nonlinearity(limb_emb)
            limb_emb = self.limb_emb_layer.dense[1](limb_emb)
        
        '''
        ##########################
            Graph Convolution
        ##########################
        '''

        condition_embedding = global_emb
        if self.local_flag == 1:
            condition_embedding += local_emb
        if self.temp_flag == 1:
            condition_embedding += past_emb
        if self.limb_flag == 1 or self.limb_flag == 2:
            temb += limb_emb

        out =  self.gconv_input2(x[:, :, -3:], self.adj) + condition_embedding

        for i in range(self.n_layers):
            out = self.atten_layers[i](out, self.src_mask)
            out = self.gconv_layers[i](out, temb, condition = condition_embedding)
            
        out = self.gconv_output(out, self.adj)

        return out, limb_len_pred 
    
class mmDiff(nn.Module):
    def __init__(self, diff_args, diff_config, 
                 feat_model = "P4Transformer",                                  # models for differernt dataset
                 global_feat_size = 64, past_frames = 6, radar_input_c = 6,      # args for different dataset
                 global_flag = 1, local_flag = 1, temp_flag = 1, limb_flag = 1
                 ):
        super(mmDiff, self).__init__()
        ### Generate Feature Encoder ###
        if feat_model == "P4Transformer":
            self.model_feat = P4Transformer_feat(radius=0.1, nsamples=32, spatial_stride=32,
                                                temporal_kernel_size=3, temporal_stride=2,
                                                emb_relu=False,
                                                dim=1024, depth=10, heads=8, dim_head=256,
                                                mlp_dim=2048, num_classes=17*3, dropout1=0.0, dropout2=0.0)
        elif feat_model == "PointTransformer":
            self.model_feat = PointTransformerReg_feat(
                                                input_dim = 5,
                                                nblocks = 5,
                                                n_p = 17)
            
        print(f"MMdiff using {feat_model} as feature extractor.")

        ### Generate Diff Model ###
        self.diff_args, self.diff_config = diff_args, diff_config
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        self.adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False).cuda()
        self.model_diff = GCNdiff(self.adj, self.diff_config, 
                                global_feat_size = global_feat_size, past_frames = past_frames, radar_input_c = radar_input_c,
                                global_flag = global_flag, local_flag = local_flag, temp_flag = temp_flag, limb_flag = limb_flag) # args for different dataset

        ### History Coarse Pose Band ###
        self.past_frames = past_frames
        # self.radar_history_list = []
        self.pose_coarse_curr = None # shape: (b, 17, 3)
        self.cemd_curr = None # shape: (b, 17, 64)
        self.x_history_list = None # shape: (b, 17, 3*(self.past_frames+1)), [[past_coarse_poses], curr_coarse_pose]
    
    def forward(self, radar, t = 0, x_curr = None, x_history = None, cemd = None): 
        

        # x_curr: shape: (b, 17, 3) for train; (b*test_time, ...) for test and inference
        # cemd: shape: (b, 17, 64) for train; (b*test_time, ...) for test and inference
        # x_history: shape: (b, 17, 3*(self.past_frames+1)), [[past_coarse_poses], curr_coarse_pose] for train; (b*test_time, ...) for test and inference

        if cemd == None and x_curr == None: 
            # Note: if cemd, x_curr == None, indicating first round of diffusion.
            x_coarse_curr, cemd_curr = self.model_feat(radar)
            x_curr, cemd = x_coarse_curr, cemd_curr
            

            # Note: if x_history == None, indicating start of inference; else x_history all provided.
            if x_history == None: 
                # if self.x_history_list is empty, initialize using coarse poses.
                x_history = torch.cat([x_coarse_curr] * self.past_frames, dim=-1)
            else:
                # update whenever new coarse pose is available
                x_history = torch.cat([x_history[...,3:],x_coarse_curr], dim = -1)
            

            
                
        x_all_list = torch.cat([x_history[...,:-3], x_curr], dim = -1)

        # Pose Diffusion
        output, limb_len_pred = self.model_diff(x_all_list, t, cemd, radar, limb_len = None) 
        # currently train and test and inference do not require gt limb length
            
        return output[:, :, -3:], x_history, cemd, limb_len_pred


class mmDiffRunner(object):
    def __init__(self, args, config, device=None, dataset = "mmBody"):
        self.args = args
        self.config = config
        self.data_time = 0
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # beta scheduling
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        if dataset == "mmBody":
            self.dataset = dataset
            self.model = mmDiff(args, config, feat_model = "P4Transformer", global_feat_size = 64, past_frames = 6, radar_input_c = 6)
        elif dataset == "MetaFi":
            self.dataset = dataset
            self.model = mmDiff(args, config, feat_model = "PointTransformer", global_feat_size = 32, past_frames = 8, radar_input_c = 5, local_flag=2, limb_flag=2)
        self.model.to(device)
        

        # For auto dataloader
        self.train_loader = None
        self.test_loader = None

        # inference step generation
        self.test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
        try:
            self.skip = self.args.skip
        except Exception:
            self.skip = 1
        
        if self.args.skip_type == "uniform":
            self.skip = test_num_diffusion_timesteps // test_timesteps
            self.seq = range(0, test_num_diffusion_timesteps, self.skip)
        elif self.args.skip_type == "quad":
            self.seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            self.seq = [int(s) for s in list(self.seq)]
        else:
            raise NotImplementedError
        
    def phase1_train(self, train_dataset, test_dataset, is_train = False, is_save = False):


        # select pretrain model
        model = self.model.model_feat
        model = model.to(self.device)

        # select dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.config.pretrain.batch_size, num_workers=self.config.pretrain.num_workers)
        train_loader_save = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=self.config.pretrain.batch_size, num_workers=self.config.pretrain.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=self.config.pretrain.batch_size, num_workers=self.config.pretrain.num_workers)

        # select optimizer

        optimizer = optim.Adam(model.parameters(),
                              lr=self.config.pretrain.learning_rate, weight_decay=self.config.pretrain.weight_decay, foreach=True)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.config.pretrain.amp)
        criterion = nn.MSELoss()


        if is_train:
            print("Trained from scratch!")
            model.train()
            for epoch in range(self.config.pretrain.n_epochs):
                
                epoch_loss = 0
                epoch_mpjpe = 0

                with tqdm(total=len(train_loader), desc=f'Train round{epoch}/{self.config.pretrain.n_epochs}', unit='batch') as pbar:
                    for data in train_loader:
                        inputs, labels = data
                        labels = labels[:,:,:] - labels[:,:1,:]
                        inputs = inputs.type(torch.FloatTensor).to(self.device)
                        labels = labels.to(self.device)
                        labels = labels.type(torch.FloatTensor)

                        optimizer.zero_grad()
                        outputs, feat = model(inputs)
                        outputs = outputs.to(self.device)
                        outputs = outputs.type(torch.FloatTensor)
                        loss = criterion(outputs, labels)
                        
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.pretrain.gradient_clipping)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                        epoch_loss += loss.item() * inputs.size(0)
                        epoch_mpjpe += (outputs - labels).square().mean().item() / labels.size(0)
                        pbar.update(1)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                epoch_loss = epoch_loss / len(train_loader.dataset)
                epoch_mpjpe = epoch_mpjpe / len(train_loader)
                print('Epoch:{}, MPJPE:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_mpjpe), float(epoch_loss)))
            # save model weights
            savepath = os.path.join(self.config.pretrain.pretrain_model_root, self.dataset + "_mmDiff_phase1.pth")
            print(f'Save model at {savepath}...')
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        else:
            pretrain_path = os.path.join(self.config.pretrain.pretrain_model_root, self.dataset + "_mmDiff_phase1.pth")
            stat = torch.load(pretrain_path)['model_state_dict']
            self.model.model_feat.load_state_dict(stat)
            model = self.model.model_feat
            print("Phase 1 use pretrained model!")


        
        if is_save:
            phase1_save(model, test_loader, self.device, save_flag = 'test', root_path=os.path.join(self.config.pretrain.save_data_root, self.dataset))
            phase1_save(model, train_loader_save, self.device, save_flag = 'train', root_path=os.path.join(self.config.pretrain.save_data_root, self.dataset))
            

        return


    def load_pretrain_dataloader(self):
        # try:
        # train_dataset = PoseGenerator_gmm(test=False, frames=self.config.model.past_frames, source_dir=self.config.pretrain.save_data_root)
        test_dataset = PoseGenerator_gmm(test=True, frames=self.model.past_frames, source_dir=os.path.join(self.config.pretrain.save_data_root, self.dataset), dataset=self.dataset)
        # except:
        #     print("Pretrain dataset is incomplete")
        train_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=self.config.training.batch_size, num_workers=self.config.training.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=self.config.training.batch_size, num_workers=self.config.training.num_workers)

        self.train_loader, self.test_loader = train_loader, test_loader

        return train_loader, test_loader


    def phase2_train(self, train_loader = None, is_train = False):
        if is_train:
            if train_loader == None: 
                if self.train_loader == None:
                    self.load_pretrain_dataloader()
                train_loader = self.train_loader
            assert train_loader != None

            cudnn.benchmark = True

            # debug
            args, config = self.args, self.config
            test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
                config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                    

            # initialize the recorded best performance
            self.best_p1, self.best_epoch = 1000, 0
            
            # initialize the optimizer
            optimizer = get_optimizer(self.config, self.model.parameters())
            
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(self.model)
            else:
                ema_helper = None
            
            start_epoch, step = 0, 0
            
            lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
        
            for epoch in range(start_epoch, self.config.training.n_epochs):
                print(f"Phase 2 training epoch {epoch}")
                data_start = time.time()
                data_time = 0
                

                # Switch to train mode
                torch.set_grad_enabled(True)
                self.model.train()
                
                epoch_loss_diff = AverageMeter()

                for i, (x_history, targets_noise_scale, cemd, targets_3d, radar, limb_len_gt) in enumerate(train_loader):
                    data_time += time.time() - data_start
                    step += 1
                    data_time += time.time() - data_start
                    # to cuda
                    x_history, targets_noise_scale, targets_3d, radar, cemd, limb_len_gt = \
                        x_history.to(self.device), targets_noise_scale.to(self.device), targets_3d.to(self.device), radar.to(self.device), cemd.to(self.device), limb_len_gt.to(self.device)
                    # x_history: ([b, 17, (past+1)*3]), *** except [b, 17, -3:] indicate gt pose for training only. need correction !
                    # targets_noise_scale: ([1024, 17, (past+1)*3]); Not implemented
                    # targets_3d: ([1024, 17, 3])
    
                    # generate nosiy sample based on seleted time t and beta
                    n = targets_3d.size(0)
                    x = targets_3d
                    e = torch.randn_like(x)
                    b = self.betas        
                    t = torch.randint(low=0, high=self.num_timesteps,
                                    size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    e = e*(targets_noise_scale)
                    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
    
                    # generate x_t (refer to DDIM equation)
                    x = x * a.sqrt() + e * (1.0 - a).sqrt()
                    
                    # predict noise
                    output_noise, _, _, limb_len_pred = self.model(radar, t = t.float(), x_curr = x, x_history = x_history, cemd = cemd)
                    loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)
                    limb_loss =  (limb_len_pred - limb_len_gt).abs().sum(dim=-1).mean(dim=0)
                    loss_diff = loss_diff + limb_loss*10

                    
                    optimizer.zero_grad()
                    loss_diff.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.optim.grad_clip)                
                    optimizer.step()
                
                    epoch_loss_diff.update(loss_diff.item(), n)
                
                    if self.config.model.ema:
                        ema_helper.update(self.model)
                    
                    if i%10 == 0 and i != 0:
                        print('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                            .format(epoch, i+1, len(self.train_loader), step, data_time, epoch_loss_diff.avg))
                        
                data_start = time.time()
                

                if epoch % decay == 0:
                    lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                    
                if epoch % 1 == 0:
                
                    print('test the performance of current model')

                    p1, p2 = self.test(is_train=True)

                    if p1 < self.best_p1:
                        self.best_p1 = p1
                        self.best_epoch = epoch

                        savepath = F"./pretrained/mmDiff_{epoch}.pth"
                        print(f'Save model at {savepath}...')
                        state = {
                            'epoch': epoch,
                            'model_state_dict': self.model.model_diff.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                        torch.save(state, savepath)
                    print('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                        .format(self.best_epoch, self.best_p1, epoch, p1, p2))
                
        else:
            pretrain_path = os.path.join(self.config.model.pretrain_model_root, self.dataset+"_mmDiff.pth")
            stat = torch.load(pretrain_path)
            # check parallel stat dict
            is_stat_parallel = False
            for k,v in stat[0].items():
                if k[:7] == "module.":
                    is_stat_parallel = True
                    break

            if is_stat_parallel:
                self.model.model_diff.load_state_dict({k.replace('module.', ''): v for k, v in stat[0].items()})
            else:
                self.model.model_diff.load_state_dict(stat[0])
            # self.model.model_feat.load(stat[0])
            print("Phase 2 use pretrained model!")

    def test(self, test_loader = None, is_train = False):
        print(f"Testing...")
        data_start = time.time()
        data_time = 0

        # Switch to test mode
        cudnn.benchmark = True
        torch.set_grad_enabled(False)
        self.model.eval()

        if test_loader == None: 
            if self.test_loader == None:
                self.load_pretrain_dataloader()
            test_loader = self.test_loader
        assert test_loader != None

        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        self.test_action_list = ["Test"]
        action_error_sum = define_error_list(self.test_action_list) 

        for i, (x_history, _, cemd, targets_3d, radar, limb_len_gt) in enumerate(test_loader):
            data_time += time.time() - data_start
            x_history, cemd, targets_3d, radar, limb_len_gt = \
            x_history.to(self.device), cemd.to(self.device), targets_3d.to(self.device), radar.to(self.device), limb_len_gt.to(self.device)

            x_history = x_history - x_history[:, :1, :] 

            # generate multiple hypothesis
            x_history = x_history.repeat(self.test_times,1,1)
            cemd = cemd.repeat(self.test_times,1,1)
            radar = radar.repeat(self.test_times,1,1,1)
            limb_len_gt = limb_len_gt.repeat(self.test_times,1)


            # Run diffusion step
            output_pose, _ = self.generalized_steps(self.seq, self.betas, self.args.eta, 
                                                radar, 
                                                x_coarse = x_history[:, :, -3:], x_history = x_history, cemd = cemd)
            output_pose = output_pose[-1]            
            output_pose = torch.mean(output_pose.reshape(self.test_times,-1,17,output_pose.shape[-1]),0)
            output_pose = output_pose - output_pose[:, :1, :]

            # evaluation code
            targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]
            epoch_loss_3d_pos.update(mpjpe(output_pose, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_pose.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))

            action_error_sum = test_calculation(output_pose, targets_3d, self.test_action_list, action_error_sum)

            p1, p2 = print_error(action_error_sum, is_train)

            if i%1 == 0 and i != 0:
                print('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(self.test_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
        print('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(batch=i + 1, size=len(self.test_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                    e2=epoch_loss_3d_pos_procrustes.avg))

        return p1, p2
    
    def inference(self, test_loader):
        # Switch to test mode
        cudnn.benchmark = True
        torch.set_grad_enabled(False)
        self.model.eval()

        x_history = None
        # Run diffusion step
        for i, (radar, trarget_3d) in enumerate(test_loader):
            step += 1
            # to cuda
            targets_3d, radar = targets_3d.to(self.device), radar.to(self.device)

            radar = radar.repeat(self.test_times,1,1,1)

            output_pose, x_history = self.generalized_steps(self.seq, self.betas, self.args.eta, 
                                                radar, 
                                                x_coarse = None, x_history = x_history, cemd = None)
            output_pose = output_pose[-1]            
            output_pose = torch.mean(output_pose.reshape(self.test_times,-1,17,output_pose.shape[-1]),0)
            output_pose = output_pose - output_pose[:, :1, :]

            # inference operation
            print(output_pose.shape)

            
        

        
        

    def generalized_steps(self, seq, b, eta, 
                          radar, 
                          x_coarse = None, x_history = None, cemd = None):
        with torch.no_grad():
            n = radar.size(0)
            x = x_coarse
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).cuda()
                next_t = (torch.ones(n) * j).cuda()
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1]
                et, x_history, cemd, limb_len_pred = self.model(radar, t = t.float(), x_curr = xt, x_history = x_history, cemd = cemd)
                # et = et + 0*self.limb_cond(xt, et, at, limb_len_pred) # 200 for gt, 50 for pred

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                x0_preds.append(x0_t)
                c1 = (
                    eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et
                xs.append(xt_next)
        return xs, x_history


    def limb_cond(self, x, et, at, limb_len):
        with torch.enable_grad():
            # calculate the limb length based on prediction
        
            limb_len = torch.autograd.Variable(limb_len, requires_grad = True)
            x = torch.autograd.Variable(x, requires_grad = True)
            # out_gra = (x - et * (1 - at).sqrt()) / at.sqrt()
            limb_list_pr = []
            limb_pairs = [[0, 1], [1, 2], [2, 3],
                                [0, 4], [4, 5], [5, 6],
                                [0, 7], [7, 8], [8, 9], [9,10],
                                [8, 11], [11, 12], [12, 13],
                                [8, 14], [14, 15], [15, 16]]
            for i in range(len(limb_pairs)):
                limb_src = limb_pairs[i][0]
                limb_end = limb_pairs[i][1]
                limb_vector_pr = x[:, limb_src, -3:] - x[:, limb_end, -3:]
                limb_list_pr.append(limb_vector_pr)

            
            out_limb = torch.stack(limb_list_pr, dim = 1)
            out_limb_len = torch.sqrt(torch.sum(torch.pow(out_limb, 2), dim=-1))

            F_limb = torch.abs((out_limb_len - limb_len)).sum(dim=-1).mean(dim=0)

            out_delta = torch.autograd.grad(F_limb, x)[0]

            return out_delta