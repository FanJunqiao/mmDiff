from __future__ import absolute_import
from lib2to3.refactor import get_fixers_from_package

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.GraFormer import *
import time



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
    def __init__(self, adj, config, time_print = False):
        super(GCNdiff, self).__init__()

        ### for time recording ###  
        self.sum_module_time = [0, 0, 0, 0, 0]
        self.iter = 0
        self.time_print = time_print
        

        self.adj = adj
        self.config = config
        ### load gcn configuration ###
        con_gcn = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, n_pts = \
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim, \
                con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
                
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim*4
                
        ### Generate Graphformer  ###
        self.n_layers = num_layers
        self.feature_size = 64
        self.past_frames = 6

        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_input1 = ChebConv(in_c=self.feature_size, out_c=self.hid_dim, K=2)
        _gconv_input2 = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
        _gconv_input3 = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
        _gconv_input_t = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
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
        self.gconv_input1 = _gconv_input1
        self.gconv_input2 = _gconv_input2
        self.gconv_input3 = _gconv_input3
        self.gconv_input_t = _gconv_input_t
        
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3 * (self.past_frames+1), K=2)
        
        
        ### diffusion configuration  ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim,self.emd_dim),
            torch.nn.Linear(self.emd_dim,self.emd_dim),
        ])

        ### self local embedding ###
        self.local_dim = self.hid_dim
        self.local_emb = nn.Module()   
        self.local_emb.dense = nn.ModuleList([
            torch.nn.Linear(6,self.local_dim // 2),
            torch.nn.Linear(self.local_dim // 2,self.local_dim),
        ])
        
        self.local_attn_iter = 5
        local_attn = MultiHeadedAttention(n_head, self.local_dim)
        _local_attention = []
        for i in range(self.local_attn_iter):
            _local_attention.append(c(local_attn))
        self.local_attention = nn.ModuleList(_local_attention)


        ### temperal embedding ###
        self.temperal_hid_dim = self.hid_dim * 17
        self.temperal_conv = nn.Sequential(
            nn.Conv1d(self.temperal_hid_dim, self.temperal_hid_dim, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(self.temperal_hid_dim, self.temperal_hid_dim, kernel_size = 3, padding = 1),
            nn.MaxPool1d(kernel_size = self.past_frames)
        )

        ### limb embedding ###
        self.limb_size = 16
        limb_feature_size = 1024
        self.limb_linear = nn.Sequential(
            torch.nn.LayerNorm(17 * self.feature_size),
            torch.nn.Linear(17 * self.feature_size,limb_feature_size),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(limb_feature_size, self.limb_size),
        )
        # self.limb_linear = nn.Sequential(
        #     torch.nn.LayerNorm((self.past_frames+1)*self.n_pts * 3),
        #     torch.nn.Linear((self.past_frames+1)*self.n_pts * 3,limb_feature_size),
        #     torch.nn.ReLU(),
        #     # torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(limb_feature_size, self.limb_size),
        # )

        self.limb_emb_layer = nn.Module()
        
        self.limb_emb_layer.dense = nn.ModuleList([
            torch.nn.Linear(self.limb_size,self.emd_dim),
            torch.nn.Linear(self.emd_dim,self.emd_dim),
        ])


    def forward(self, x, mask, t, cemd, radar, limb_len):

        start_time = time.time() # time recording

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
        
        global_start_time = time.time()
        global_emb = self.gconv_input1(cemd, self.adj)
        global_end_time = time.time()
        global_time = (global_end_time - global_start_time)


        '''
        ##########################
            Local Embedding
        ##########################
        '''
        local_start_time = time.time()
        thre = 0.04
        local_emb_list = []
        point_cloud = radar.view(-1, radar.size()[1] * radar.size()[2], 6)
        # point_cloud = radar[:, 0, :, :]

        # # debug
        # from matplotlib import pyplot as plt
        # import os
        # fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex=False, sharey = False)
        # fig.set_figwidth(10)
        # all_p = point_cloud.detach().cpu().numpy()
        # axes[0].scatter(all_p[0, :, 0], all_p[0, :, 2],marker="o", c="b")
        # axes[1].scatter(all_p[0, :, 0], all_p[0, :, 1],marker="o", c="b")


        for joint_id in range(17):
            # select local 50 points, with 0 paddings
            anchor = x[:, [joint_id], -3:]
            dist = torch.sum(torch.pow(point_cloud[:, :, :3] - anchor, 2), dim=-1)
            index_sort = torch.argsort(dist, dim = -1).unsqueeze(-1).expand(-1,-1,6)
            point_cloud_select = torch.gather(point_cloud, 1, index_sort)[:,:50,:]

            # point_cloud_select[:, :, :3] -= anchor # normalize
            
            dist_sort = torch.sum(torch.pow(point_cloud_select[:,:,:3] - anchor, 2), dim=-1)
            dist_scale = torch.sum(dist_sort.lt(thre), dim=-1, keepdim = True) / 50


            
        #    # debug
        #     color_p = point_cloud_select.detach().cpu().numpy()
        #     anchor_p = anchor.detach().cpu().numpy()
            
        #     axes[0].scatter(color_p[0, :, 0], color_p[0, :, 2],marker="x", c="r")
            
        #     axes[1].scatter(color_p[0, :, 0], color_p[0, :, 1],marker="x", c="r")
        #     Drawing_uncolored_circle = plt.Circle( (anchor_p[0,0,0], anchor_p[0,0,2]),
        #                               0.08 ,
        #                               fill = False )
        #     axes[0].add_artist( Drawing_uncolored_circle )
        #     axes[0].set_xlim([-1.6, 1.6])
        #     axes[0].set_ylim([-1.6, 1.6])
        #     Drawing_uncolored_circle = plt.Circle( (anchor_p[0,0,0], anchor_p[0,0,1]),
        #                               0.08 ,
        #                               fill = False )
        #     axes[1].add_artist( Drawing_uncolored_circle )
        #     axes[1].set_xlim([-1.6, 1.6])
        #     axes[1].set_ylim([-1.6, 1.6])
        #     # debug
            



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
        local_emb = torch.stack(local_emb_list, dim=1)
        local_end_time = time.time()
        local_time = (local_end_time - local_start_time)
        # # debug
        # plt.title(f"PCK0.1 = {dist_scale[0]}")
        # plt.savefig(os.path.join("debug", f"prediction_{100}.png"))
        # plt.close()


          
        
        '''
        ##########################
            Temperal Embedding
        ##########################
        '''
        temp_start_time = time.time()
        
        x_past_list = []
        # out = self.gconv_input(x, self.adj)
        for i in range(self.past_frames):
            x_curr = x[:, :, i*3:(i+1)*3]
            x_curr = self.gconv_input_t(x_curr, self.adj)   
            x_past_list.append(x_curr)   

        x_past = torch.stack(x_past_list, dim=1)
        past_emb = x_past.view(-1, x_past.size(1), x_past.size(2) * x_past.size(3)).permute(0,2,1)
        past_emb = self.temperal_conv(past_emb).squeeze().view(-1, x_past.size(2), x_past.size(3))


        temp_end_time = time.time()
        temp_time = temp_end_time - temp_start_time

        '''
        ##########################
            Limb Embedding
        ##########################
        '''
        limb_start_time = time.time()
        limb_len_pred = self.limb_linear(cemd.view(-1, 17 * self.feature_size))
        # limb_len_pred = self.limb_linear(x.reshape(-1, (self.past_frames+1)*self.n_pts * 3))

        if limb_len == None:
            limb_len = limb_len_pred

        limb_emb = self.limb_emb_layer.dense[0](limb_len)
        limb_emb = nonlinearity(limb_emb)
        limb_emb = self.limb_emb_layer.dense[1](limb_emb)

        limb_end_time = time.time()
        limb_time = limb_end_time - limb_start_time
        '''
        ##########################
            Graph Convolution
        ##########################
        '''
        condition_embedding =    local_emb + global_emb + past_emb
        # condition_embedding = self.gconv_input3(x[:, :, -3:], self.adj)
        
        out =  self.gconv_input2(x[:, :, -3:], self.adj) + condition_embedding
        # out = self.gconv_input3(x, self.adj)

        for i in range(self.n_layers):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out, temb+limb_emb, condition = condition_embedding)
            
        out = self.gconv_output(out, self.adj)


        real_iter = 1
        end_time = time.time()
        overall_time = end_time - start_time
        if (self.iter > 10):
            

            self.sum_module_time[0] += global_time 
            self.sum_module_time[1] += local_time
            self.sum_module_time[2] += limb_time
            self.sum_module_time[3] += temp_time
            self.sum_module_time[4] += overall_time
            real_iter = self.iter - 10
        self.iter += 1
        


        if self.time_print == True:
            print(f"global: {self.sum_module_time[0]*1000/real_iter}; local: {self.sum_module_time[1]*1000/real_iter}; limb: {self.sum_module_time[2]*1000/real_iter}; temp: {self.sum_module_time[3]*1000/real_iter}; all: {self.sum_module_time[4]*1000/real_iter};")
        return out, limb_len_pred 
    
