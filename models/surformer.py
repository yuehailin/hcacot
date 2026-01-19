from torch.nn import init

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from utils.core_utils import *

def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim=512, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, query_dim=256, context_dim=256, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, context):
        h = self.heads

        q = self.to_q(x1)
        k, v = self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim_ori = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim_ori.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out), sim_ori


class Self_Attention(nn.Module):
    def __init__(self, query_dim=256, context_dim=256, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, mask=None):
        h = self.heads

        q = self.to_q(x1)
        k, v = self.to_k(x1), self.to_v(x1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant_(m.bias.data, 0.0)


class MIL_Sum_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Deep Sets Implementation.

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Sum_FC_surv, self).__init__()
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}

        # Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(self.phi, device_ids=device_ids).to('cuda:0')

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        h_path = self.phi(x_path).sum(axis=0)
        h_path = self.rho(h_path)
        h = h_path  # [256] vector

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, h, None


class surformer(nn.Module):
    def __init__(self,config, size_arg="small", dropout=0.25, n_classes=4):
        super(surformer, self).__init__()

        self.num_prototype = 5
        self.dropout_rate = 0.25
        self.c_local = 128
        self.c_global = 256
        self.p_threshold = 1 / (self.num_prototype-1)

        self.prototype = nn.Parameter(torch.randn((1, self.num_prototype, 256), requires_grad=True))

        self.non_lin = nn.Sequential(nn.Linear(512, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(256, 256))

        self.ln_0 = nn.LayerNorm(256)
        self.ln_1 = nn.LayerNorm(256)
        self.ln_2 = nn.LayerNorm(256)
        self.ln_3 = nn.LayerNorm(256)
        self.ln_4 = nn.LayerNorm(256)
        self.ln_5 = nn.LayerNorm(256)
        self.ln_6 = nn.LayerNorm(256)
        self.ln_7 = nn.LayerNorm(256)
        self.ln_8 = nn.LayerNorm(256)
        self.ln_local = nn.LayerNorm((self.num_prototype - 1) * self.c_local)

        self.cross_attention_0 = Cross_Attention(dropout=self.dropout_rate)
        self.cross_attention_1 = Cross_Attention(dropout=self.dropout_rate)

        self.self_attention_0 = Self_Attention(dropout=self.dropout_rate)

        self.ffn_0 = FeedForward(dim=256, dropout=self.dropout_rate)
        self.ffn_1 = FeedForward(dim=256, dropout=self.dropout_rate)
        self.ffn_2 = FeedForward(dim=256, dropout=self.dropout_rate)
        self.ffn_3 = FeedForward(dim=256, dropout=self.dropout_rate)

        self.compress_local = nn.Sequential(nn.Linear(256, self.c_local),
                                            nn.LayerNorm(self.c_local),
                                            nn.ReLU(inplace=True))

        self.classifier_global = nn.Linear(
            self.c_global, n_classes, bias=False)
        self.classifier_local = nn.Linear(
            self.c_local*(self.num_prototype-1), n_classes, bias=False)
        self.classifier_overall = nn.Linear(
            self.c_local * (self.num_prototype - 1) + self.c_global, n_classes, bias=False)

        # self.classifier_global = nn.Linear(self.c_global, n_classes)
        # self.classifier_local = nn.Linear(self.c_local * (self.num_prototype - 1), n_classes)
        # self.classifier_overall = nn.Linear(self.c_local * (self.num_prototype - 1) + self.c_global, n_classes)

        self.compress_local.apply(weights_init_kaiming)
        self.classifier_global.apply(weights_init_classifier)
        self.classifier_local.apply(weights_init_classifier)
        self.classifier_overall.apply(weights_init_classifier)
        self.hard_or_soft = config.hard_or_soft

    def forward_global(self, x):
        logits = self.classifier_global(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def forward_local(self, x):
        logits = self.classifier_local(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def forward_overall(self, x):
        logits = self.classifier_overall(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def forward(self, x_s, coord_s, x_l, coords_l, label, staus,time,disc,soft_0, soft_1, soft_2, soft_3):
        x_path = x_s
        x_path = x_path.unsqueeze(0) 
        x_ori = self.non_lin(x_path)

        x, attn = self.cross_attention_0(self.ln_0(self.prototype), self.ln_1(x_ori))

        # for weight generation
        attn = attn[:, 1:, :]
        attn = F.softmax(attn, dim=1)
        value, attn = torch.max(attn, dim=1)
        value = (value > self.p_threshold).int()

        attn_weight = []
        for i in range(self.num_prototype-1):
            temp = ((attn == i).int()*value).sum()
            attn_weight.append(temp[None, ])

        attn_weight = torch.cat(attn_weight, dim=0)
        attn_weight = attn_weight / value.sum()

        # print(attn_weight)

        x = self.ffn_0(self.ln_2(x)) + x

        x = self.self_attention_0(self.ln_3(x)) + x
        x = self.ffn_1(self.ln_4(x)) + x

        x = self.cross_attention_0(self.ln_0(self.prototype), self.ln_5(x))[0] + x
        x = self.ffn_2(self.ln_6(x)) + x

        x_global, x_local = x[:, 0, :][:, None, :], x[:, 1:, :]
        x_local = (attn_weight[None, :, None] + 1) * x_local
        x_local = x_local.view(1, self.num_prototype-1, 256)

        x = torch.cat((x_global, x_local), dim=1)
        x = self.self_attention_0(self.ln_7(x)) + x
        x = self.ffn_3(self.ln_8(x)) + x

        x_global, x_local = x[:, 0, :], x[:, 1:, :]

        x_local = self.compress_local(x_local).view(1, -1) #[N, 128]
        x_local = self.ln_local(x_local)

        x = torch.cat((x_global, x_local), dim=-1)
        logits = self.classifier_overall(x)

        Y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

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

if __name__ == "__main__":
    input_data = torch.randn((1, 4900, 512))
    model = surformer()
    output = model(input_data)
    print(output)