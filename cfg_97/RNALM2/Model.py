from numpy import select
import torch
from torch import nn
from torch.nn import functional as F
from RNALM2 import basic,Evoformer
import math,sys
from torch.utils.checkpoint import checkpoint
import numpy as np


def one_d(idx_, d, max_len=2056*8):
    idx = idx_[None]
    K = torch.arange(d//2).to(idx.device)
    sin_e = torch.sin(idx[..., None] * math.pi / (max_len**(2*K[None]/d))).to(idx.device)
    cos_e = torch.cos(idx[..., None] * math.pi / (max_len**(2*K[None]/d))).to(idx.device)
    return torch.cat([sin_e, cos_e], axis=-1)[0]






class RNAembedding(nn.Module):
    def __init__(self,cfg):
        super(RNAembedding,self).__init__()
        self.s_in_dim=cfg['s_in_dim']
        self.z_in_dim=cfg['z_in_dim']
        self.s_dim=cfg['s_dim']
        self.z_dim=cfg['z_dim']
        self.qlinear  =basic.Linear(self.s_in_dim+1,self.z_dim)
        self.klinear  =basic.Linear(self.s_in_dim+1,self.z_dim)
        self.slinear  =basic.Linear(self.s_in_dim+1,self.s_dim)
        self.zlinear  =basic.Linear(self.z_in_dim+1,self.z_dim)

        self.poslinears = basic.Linear(64,self.s_dim)
        self.poslinearz = basic.Linear(64,self.z_dim)
    def forward(self,in_dict):
        # msa N L D, seq L D
        # mask: maksing, L, 1 means masked
        # aa:   L x s_in_dim
        # ss:   L x L x 2
        # idx:  L (LongTensor)
        L = in_dict['aa'].shape[0]
        aamask = in_dict['mask'][:,None]
        zmask = in_dict['mask'][:,None] + in_dict['mask'][None,:]
        zmask[zmask>0.5]=1
        zmask = zmask[...,None]
        s = torch.cat([aamask,(1-aamask)*in_dict['aa']],dim=-1)
        sq=self.qlinear(s)
        sk=self.klinear(s)
        z=sq[None,:,:]+sk[:,None,:]
        seq_idx = in_dict['idx'][None]
        relative_pos = seq_idx[:, :, None] - seq_idx[:, None, :]
        relative_pos = relative_pos.reshape([1, -1])
        relative_pos =one_d(relative_pos,64)
        z = z + self.poslinearz( relative_pos.reshape([1, L, L, -1])[0] )

        s = self.slinear(s) + self.poslinears( one_d(in_dict['idx'], 64)   )
        
        return s,z


class RNA2nd(nn.Module):
    def __init__(self,cfg):
        super(RNA2nd,self).__init__()
        self.s_in_dim=cfg['s_in_dim']
        self.z_in_dim=cfg['z_in_dim']
        self.s_dim=cfg['s_dim']
        self.z_dim=cfg['z_dim']
        self.N_elayers =cfg['N_elayers']
        self.emb    = RNAembedding(cfg)
        self.evmodel=Evoformer.Evoformer(self.s_dim,self.z_dim,self.N_elayers)   
        self.seq_head = basic.Linear(self.s_dim,self.s_in_dim) 
        self.joint_head = basic.Linear(self.z_dim,self.s_in_dim*self.s_in_dim)




    def embedding(self,in_dict):
        s,z = self.emb(in_dict)
        s,z = self.evmodel(s[None,...],z)
        return s[0],z        









