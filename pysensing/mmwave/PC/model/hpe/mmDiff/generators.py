import numpy as np
import torch
from torch.utils.data import Dataset
import os


class PoseGenerator_gmm(Dataset):
    def __init__(self, test = False, frames = 8, source_dir = "", dataset = "mmBody"):

        self.source_dir = source_dir
        self.test = test
        self.frames = frames


        if self.test == False:
            with open(os.path.join(self.source_dir, "joint_gt_array_train.npy"), 'rb') as f:
                self._poses_3d_mm = np.load(f) # list of 3d gt, with (a, j, [x, y, z])
            with open(os.path.join(self.source_dir, "joint_pred_array_train.npy"), 'rb') as f:
                self._poses_3d_pr_mm = np.load(f) # list of 3d gt, with (a, j, [x, y, z])
            with open(os.path.join(self.source_dir, "radar_feat_array_train.npy"), 'rb') as f:
                self._feat_mm = np.load(f) # list of 2d gmm model, with (a, j, kernel_n, [pi, ux, uy, vx, vy])

            self._radar_mm = []
            if dataset == "mmBody":
                for i in range(5):
                    with open(os.path.join(self.source_dir, f"radar_input_array_train{i}.npy"), 'rb') as f:
                        self._radar_mm.append(np.load(f))
                self._radar_mm = np.concatenate(self._radar_mm)
            elif dataset == "MetaFi":
                self._radar_mm = np.zeros((self._feat_mm.shape[0], 50, 5))



        else:
            with open(os.path.join(self.source_dir, "joint_gt_array_test.npy"), 'rb') as f:
                self._poses_3d_mm = np.load(f) # list of 3d gt, with (a, j, [x, y, z])
            with open(os.path.join(self.source_dir, "joint_pred_array_test.npy"), 'rb') as f:
                self._poses_3d_pr_mm = np.load(f) # list of 3d gt, with (a, j, [x, y, z])
            with open(os.path.join(self.source_dir, "radar_feat_array_test.npy"), 'rb') as f:
                self._feat_mm = np.load(f) # list of 2d gmm model, with (a, j, kernel_n, [pi, ux, uy, vx, vy])

            if dataset == "mmBody":
                with open(os.path.join(self.source_dir, "radar_input_array_test.npy"), 'rb') as f:
                    self._radar_mm = np.load(f) # list of 2d gmm model, with (a, j, kernel_n, [pi, ux, uy, vx, vy])
            elif dataset == "MetaFi":
                self._radar_mm = np.ones((self._feat_mm.shape[0], 50, 5))



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

        
        # pose normalization
        self._poses_3d_mm[:,:,:] = self._poses_3d_mm[:,:,:]-self._poses_3d_mm[:,[0],:]
        self._poses_3d_pr_mm[:,:,:] = self._poses_3d_pr_mm[:,:,:]-self._poses_3d_pr_mm[:,[0],:]

        print('Generating {} poses...'.format(self._poses_3d_mm.shape[0]))

    def __getitem__(self, index):


        out_pose_3d = self._poses_3d_mm[index]
        out_pose_3d_pr = self._poses_3d_pr_mm[index]
        out_pose_feat = self._feat_mm[index]
        out_radar = self._radar_mm[index]
        out_limb = self._limb_3d_mm[index]

        out_radar = [np.expand_dims(self._radar_mm[index], axis=0)]
        for i in range(4-1):
            index_curr = index - 4 + i
            if index_curr < 0: index_curr = index + i
            out_radar.append(np.expand_dims(self._radar_mm[index_curr], axis=0))
        out_radar = np.concatenate(out_radar)

        out_pose_noise_scale = np.ones(out_pose_3d.shape)


        if self.test:
            out_pose_pr_curr = out_pose_3d_pr
        else:
            out_pose_pr_curr = out_pose_3d

        out_pose_pr_list = []
        
        for i in range(self.frames):
            index_curr = index - self.frames + i
            if index_curr < 0: index_curr = index + i
            past_out_pose_pr = self._poses_3d_pr_mm[index_curr]
            out_pose_pr_list.append(past_out_pose_pr)        
        out_pose_pr_list.append(out_pose_pr_curr)
        out_pose_pr = np.concatenate(out_pose_pr_list, axis=1)

        

        out_pose_pr = torch.from_numpy(out_pose_pr).float()
        out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale).float()
        out_pose_feat = torch.from_numpy(out_pose_feat).float()
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_radar = torch.from_numpy(out_radar).float()
        out_limb = torch.from_numpy(out_limb).float()

        
        return out_pose_pr, out_pose_noise_scale, out_pose_feat, out_pose_3d, out_radar, out_limb

    def __len__(self):
        return self._poses_3d_mm.shape[0]