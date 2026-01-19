# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from numpy.core.fromnumeric import squeeze
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from nystrom_attention import Nystromformer
from timm.models.layers import trunc_normal_
# x,residual  [B,C,H,W]
from utils.core_utils import *

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PEG(nn.Module):
    def __init__(self, dim=256, k=7):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)


    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


# It's challenging for TransMIL to process all the high-dimensional data in the patient-level bag, so we reduce the dimension from 1024 to 128.
class TransMIL(nn.Module):
    def __init__(self, config, num_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PEG(128)
        self._fc1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=128)
        self.layer2 = TransLayer(dim=128)
        self.norm = nn.LayerNorm(128)
        self._fc2 = nn.Linear(128, self.n_classes)
        self.hard_or_soft = config.hard_or_soft


    def forward(self, x_s, coord_s, x_l, coords_l, x_f, label, staus,time,disc,soft_0, soft_1, soft_2, soft_3):

        h = x_s
        # print(h.shape)
        # print('label',label.shape)
        # print('x_l',x_l.shape)
        # print('x_s',x_s.shape)
        
        h = h.unsqueeze(0) 
        # print(h.shape)
        #---->Dimensionality reduction first
        h = self._fc1(h) #[B, n, 128]
        
        #---->padding
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 128]

        #---->Add position encoding, after a transformer
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 128]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 128]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 128]
        # print(h.shape)
        h = self.norm(h)[:,0]
        # print(h.shape)

        #---->predict output
        logits = self._fc2(h)
        # print(logits.shape)


        disc = disc.unsqueeze(1)
        staus = staus.unsqueeze(1) 
 

        if self.hard_or_soft:    #  True 使用软标签
            loss = nll_loss_soft(logits,disc,staus, soft_0, soft_1, soft_2, soft_3, alpha=0.4,eps=1e-7, reduction='mean')
        else:
            loss = nll_loss(logits,disc,staus,alpha=0.4,eps=1e-7, reduction='mean')
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
        
        return logits,Y_hat,loss

