from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce
import os


class PoseGenerator_gmm(Dataset):
    def __init__(self, poses_3d, poses_2d_gmm, actions, camerapara, test = False, frames = 6):
        assert poses_3d is not None
        datasource = "mmBody"

        

        self._poses_3d = np.concatenate(poses_3d) # list of 3d gt, with (a, j, [x, y, z])
        self._poses_2d_gmm = np.concatenate(poses_2d_gmm) # list of 2d gmm model, with (a, j, kernel_n, [pi, ux, uy, vx, vy])
        self._actions = reduce(lambda x, y: x + y, actions)
        self._camerapara = np.concatenate(camerapara)
        self._kernel_n = self._poses_2d_gmm.shape[2]
        self.test = test
        self.frames = frames
        self.action_list = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]


        if test == False:
            with open(os.path.join(datasource, "joint_gt_array_train.npy"), 'rb') as f:
                self._poses_3d_mm = np.load(f) # list of 3d gt, with (a, j, [x, y, z])
            with open(os.path.join(datasource, "joint_pred_array_train.npy"), 'rb') as f:
                self._poses_3d_pr_mm = np.load(f) # list of 3d gt, with (a, j, [x, y, z])
            with open(os.path.join(datasource, "radar_feat_array_train.npy"), 'rb') as f:
                self._feat_mm = np.load(f) # list of 2d gmm model, with (a, j, kernel_n, [pi, ux, uy, vx, vy])
            with open(os.path.join(datasource, "sequence_array_train.npy"), 'rb') as f:
                self._sequence_mm = np.load(f) 
            self._radar_mm = []
            for i in range(5):
                with open(os.path.join(datasource, f"radar_input_array_train{i}.npy"), 'rb') as f:
                    self._radar_mm.append(np.load(f))
            self._radar_mm = np.concatenate(self._radar_mm)
            self._actions_mm = np.asarray(["Train"] * self._poses_3d_mm.shape[0])
        else:
            self._poses_3d_mm = []
            self._poses_3d_pr_mm = []
            self._feat_mm  = []
            self._sequence_mm = []
            self._radar_mm = []
            self._actions_mm = []
            for test_scene in self.action_list:
                

                test_scene_len = 0
                with open(os.path.join(datasource, f"joint_gt_array_test_{test_scene}.npy"), 'rb') as f:
                    poses_3d_mm = np.load(f)
                    self._poses_3d_mm.append(poses_3d_mm) # list of 3d gt, with (a, j, [x, y, z])
                    test_scene_len = poses_3d_mm.shape[0]
                with open(os.path.join(datasource, f"joint_pred_array_test_{test_scene}.npy"), 'rb') as f:
                    self._poses_3d_pr_mm.append(np.load(f)) # list of 3d gt, with (a, j, [x, y, z])
                with open(os.path.join(datasource, f"radar_feat_array_test_{test_scene}.npy"), 'rb') as f:
                    self._feat_mm.append(np.load(f)) # list of 2d gmm model, with (a, j, kernel_n, [pi, ux, uy, vx, vy])
                with open(os.path.join(datasource, f"sequence_array_test_{test_scene}.npy"), 'rb') as f:
                    self._sequence_mm.append(np.load(f))
                with open(os.path.join(datasource, f"radar_input_array_test_{test_scene}.npy"), 'rb') as f:
                    self._radar_mm.append(np.load(f))
                self._actions_mm.append(np.asarray([test_scene] * test_scene_len))
            self._poses_3d_mm = np.concatenate(self._poses_3d_mm)
            self._poses_3d_pr_mm = np.concatenate(self._poses_3d_pr_mm)
            self._feat_mm = np.concatenate(self._feat_mm)
            self._sequence_mm = np.concatenate(self._sequence_mm)
            self._radar_mm = np.concatenate(self._radar_mm)
            self._actions_mm = np.concatenate(self._actions_mm)

        
        #######
        # Generating Limb list
        ######

          
        limb_list = []
        limb_pairs = [[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]]
        for i in range(len(limb_pairs)):
            limb_src = limb_pairs[i][0]
            limb_end = limb_pairs[i][1]
            limb_vector = self._poses_3d_mm[:, limb_src, :] - self._poses_3d_mm[:, limb_end, :]
            limb_list.append(limb_vector)


        
        self._limb_3d_mm = np.stack(limb_list, axis = 1)
        self._limb_3d_mm = np.sqrt(np.sum(np.power(self._limb_3d_mm, 2), axis=-1))


    

        #########

        
        
        self._camerapara_mm = self._camerapara[:self._poses_3d_mm.shape[0]]

        self._poses_3d[:,:,:] = self._poses_3d[:,:,:]-self._poses_3d[:,:1,:]

        self.loader_len = self._radar_mm.shape[0]

        assert self._poses_3d.shape[0] == self._poses_2d_gmm.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions_mm)))

    def __getitem__(self, index):

        out_pose_3d = self._poses_3d_mm[index]
        out_pose_3d_pr = self._poses_3d_pr_mm[index]
        out_pose_feat = self._feat_mm[index]
        out_action = self._actions_mm[index]
        out_camerapara = self._camerapara_mm[index]
        out_radar = [np.expand_dims(self._radar_mm[index], axis=0)]
        sequence = self._sequence_mm[index]
        for i in range(4-1):
            index_curr = index - 4 + i
            if index_curr < 0 or self._sequence_mm[index_curr] != sequence: index_curr = index + i
            out_radar.append(np.expand_dims(self._radar_mm[index_curr], axis=0))
        out_radar = np.concatenate(out_radar)
        out_limb = self._limb_3d_mm[index]




        if self.test:
            out_pose_uvxyz_curr = out_pose_3d_pr
        else:
            out_pose_uvxyz_curr = out_pose_3d
        out_pose_noise_scale_curr = np.ones(out_pose_3d.shape) 

        out_pose_uvxyz_list = []
        out_pose_noise_scale_list = []
        for i in range(self.frames):
            index_curr = index - self.frames + i
            if index_curr < 0 or self._sequence_mm[index_curr] != sequence: index_curr = index
            if self.test:
                past_out_pose_uvxyz = self._poses_3d_pr_mm[index_curr]
            else:
                past_out_pose_uvxyz = self._poses_3d_pr_mm[index_curr]
            out_pose_uvxyz_list.append(past_out_pose_uvxyz)

            past_out_pose_noise_scale = np.ones(past_out_pose_uvxyz.shape) * 0.0
            out_pose_noise_scale_list.append(past_out_pose_noise_scale)
        
        out_pose_uvxyz_list.append(out_pose_uvxyz_curr)
        out_pose_noise_scale_list.append(out_pose_noise_scale_curr)

        out_pose_uvxyz = np.concatenate(out_pose_uvxyz_list, axis=1)
        out_pose_noise_scale = np.concatenate(out_pose_noise_scale_list, axis=1)


        

        out_pose_uvxyz = torch.from_numpy(out_pose_uvxyz).float()
        out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale).float()
        out_pose_feat = torch.from_numpy(out_pose_feat).float()
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_camerapara = torch.from_numpy(out_camerapara).float()
        out_radar = torch.from_numpy(out_radar).float()
        out_limb = torch.from_numpy(out_limb).float()

        
        return out_pose_uvxyz, out_pose_noise_scale, out_pose_feat, out_pose_3d, out_action, out_camerapara, out_radar, out_limb

    def __len__(self):
        return len(self._actions_mm)