import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.core_utils import *
#----> Attention module
class Attn_Net(nn.Module):
    
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 4):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

#----> Attention Gated module
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 256, D = 128, dropout = False, n_classes = 4):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.norm = nn.LayerNorm(D)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x) #[N, 256]
        b = self.attention_b(x) #[N, 256]
        A = a.mul(b) #torch.mul(a, b)
        A = self.norm(A)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class AMIL(nn.Module):
    def __init__(self, config, n_classes, gate = False):
        super(AMIL, self).__init__()
        fc = [nn.Linear(512,256), nn.ReLU()] #1024->512
        if gate:
            attention_net = Attn_Net_Gated(L = 256, D = 128, n_classes = 1)
        else:
            attention_net = Attn_Net(L = 256, D = 128, n_classes = 4)
        
        self.hard_or_soft = config.hard_or_soft
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(256, n_classes)

    def forward(self, x_s, coord_s, x_l, coords_l, label, staus, time, disc,soft_0, soft_1, soft_2, soft_3):
        
        h = x_l
        
        #---->Attention
        A, h = self.attention_net(h)  # NxK     

        

        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N


        h = torch.mm(A, h) 

        #---->predict output
        logits = self.classifiers(h) #[B, n_classes]

        disc = disc.unsqueeze(1)
        staus = staus.unsqueeze(1)


        if self.hard_or_soft:    #  Trwe 使用软标签
            loss = nll_loss_soft(logits,disc,staus, soft_0, soft_1, soft_2, soft_3, alpha=0.4,eps=1e-7, reduction='mean')
        else:
            loss = nll_loss(logits,disc,staus,alpha=0.4,eps=1e-7, reduction='mean')
        

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}

       

        return logits,Y_hat,loss













