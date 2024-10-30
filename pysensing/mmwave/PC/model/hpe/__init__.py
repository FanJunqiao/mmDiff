from .p4trans.point_4d_convolution import *
from .p4trans.transformer import * # alias for models' transformer
from .pointTrans.mmwave_point_transformer import *
from .pointTrans.transformer import *
from .mmDiff import mmDiffRunner, load_mmDiff

import os
import torch
import torch.nn.functional as F
from torch import nn




class PointTransformerReg(nn.Module):
    r"""
    PointTransformer implementation of human pose estimation (hpe), for dataset "MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset".

    args:
        input_dim (int): Feature dimension of each point.

        nblocks (int): Number of point transformer blocks.

        nneighbor (int): Number of neighbours for each point anchor.

        transformer_dim (int): Inner feature dimension within each transformer block.

        n_p (int): Number of estimated keypoints/joints.

    """
    def __init__(self, input_dim = 5, nblocks = 5, nneighbor = 16, transformer_dim = 128, n_p = 17):
        super().__init__()
        self.backbone = Backbone(
            nblocks = nblocks,
            nneighbor = nneighbor,
            input_dim = input_dim,
            transformer_dim = transformer_dim
        )
        
        self.nblocks = nblocks
        self.n_p = n_p

        dim, depth, heads, dim_head, mlp_dim, dropout = 512, 5, 4, 128, 256, 0.0
        self.joint_posembeds_vector = nn.Parameter(torch.tensor(self.get_positional_embeddings1(self.n_p, dim)).float())
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)

        mid_dim = 32
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, mid_dim),

        )

        self.fc3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(mid_dim, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 3),

        )



    
    def forward(self, x):
        f'''
        The model receive aggregated point clouds of 5 frames (referring to mm-Fi dataset), and output the 3d coordinate of n_p (n_p = 17) joints/keypoints.

        Input:
            Aggregated point clouds according to mm-Fi dataset, with preprocessing: (1) aggregation (n_frame = 5, stride = 1) (2) padding (n_points = 150)

        Output:
            3d coordinates of n_p keypoints/joints (tensor): Of shape (b, n_p, 3).

        args:
            x (tensor): Aggregated point clouds of shape (b, 5, 150, input_dim)

        '''
        if len(x.shape) == 4: 
            b, t, n, c = x.shape
            x = x = x.view(b, t*n, c)
        points, _ = self.backbone(x)
        joint_embedding =  self.joint_posembeds_vector.expand(b, -1, -1) # torch.rand(size = (points.size()[0], self.n_p, points.size()[2])).cuda() +
        embedding = torch.cat([joint_embedding, points], dim=1)
        output = self.transformer(embedding)[:, :self.n_p, :]

        feat = self.fc2(output)
        pts = self.fc3(feat)

        
        return pts
    
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


class P4Transformer(nn.Module):
    r"""
    P4Transformer implementation of human pose estimation (hpe), for dataset "mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radarr".

    args:
        radius (float): Param for Point 4D convolution. Radius of point anchor for point grouping. Default 0.1.

        nsamples (int): Param for Point 4D convolution. Number of points for each point anchor. Default 32.

        spatial_stride (int): Param for Point 4D convolution. Spatial stride for each point anchor. Default 3.

        temporal_kernel_size (int): Param for Point 4D convolution. Temporal window size for each frame anchor. Default 3.

        temporal_stride (int): Param for Point 4D convolution. Temporal stride for each frame anchor. Default 2.

        emb_relu (int): Whether using relu embedding. Default False.

        dim (int): Param for Transformer. Feature dimension of transformerf. Default 1024.

        depth (int): Param for Transformer. Depth of transformer. Default 10.

        heads (int): Param for Transformer. Heads of transformer. Default 8.

        dim_head (int): Param for MLP head. Feature dimension of MLP head. Default 256.

        num_classes (int): Output dimension. Default 17*3.

        dropout1 (float): Dropout rate between [0, 1] for Transformer.

        dropout2 (float): Dropout rate between [0, 1] for MLP head.


    """
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes,                                                  # output
                 dropout1=0.0, dropout2=0.0,                                            # dropout
                ):                                           
        super().__init__()




        self.tube_embedding = P4DConv(in_planes=3, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            # nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )
        self.joint_num = 17

        self.joint_posembeds_vector = nn.Parameter(torch.tensor(self.get_positional_embeddings1(self.joint_num, 1024)).float())
        # point Prediction Head
        input_dim = dim
        mid_dim = 64
        

        self.dim_reduce_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, mid_dim)
        )
        
        input_dim = mid_dim
        self.point_prediction_heads = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 3)
        )

        self.var_prediction_heads = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 3),
        )



    def forward(self, radar): 
        f'''
        The model receive aggregated point clouds of 4 frames (referring to mmBody dataset), and output the 3d coordinate of n_p (n_p = 17) joints/keypoints.

        Input:
            Aggregated point clouds according to mmBody dataset, with preprocessing: (1) aggregation (n_frame = 4, stride = 1) (2) padding (n_points = 5000)

        Output:
            3d coordinates of n_p keypoints/joints (tensor): Of shape (b, num_classes//3, 3).

        args:
            radar (tensor): Aggregated point clouds of shape (b, 4, 5000, input_dim=6)

        '''
        point_cloud = radar[:, :, :, :3]
        point_fea = radar[:, :, :, 3:].permute(0, 1, 3, 2)                                                                                                           # [B, L, N, 3]
        device = radar.get_device()
        
        xyzs, features = self.tube_embedding(point_cloud, point_fea)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        
        xyzts_embd = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts_embd + features
        

        if self.emb_relu:
            embedding = self.emb_relu(embedding)


        # For max-pooling algorithm
        output = self.transformer(embedding)

        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        
        output = self.mlp_head(output).view(-1, 17, 3)

        return output
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
    



def load_hpe_model(dataset, model_name):
    if model_name == 'P4Transformer':
        model = P4Transformer(radius=0.1, nsamples=32, spatial_stride=32,
                  temporal_kernel_size=3, temporal_stride=2,
                  emb_relu=False,
                  dim=1024, depth=10, heads=8, dim_head=256,
                  mlp_dim=2048, num_classes=17*3, dropout1=0.0, dropout2=0.0)
    elif model_name == 'PointTransformer':
        model = PointTransformerReg(
                    input_dim = 5,
                    nblocks = 5,
                    n_p = 17
                )
    elif model_name == "mmDiff":
        model = load_mmDiff(dataset=dataset)
        
    else:
        raise ValueError("Unsupported model. Please choose from 'P4Transformer'.")
    return model


def load_hpe_pretrain(model, pretrain_root, dataset, model_name):
    pretrain_path = os.path.join(pretrain_root + dataset, dataset + '_' + model_name + '.pt')
    stat = torch.load(pretrain_path)
    model.load_state_dict(stat)

    return model