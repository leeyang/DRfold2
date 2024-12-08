import torch
from torch import nn
from torch.nn import functional as F
from RNALM2 import basic,EvoPair,EvoMSA
import math,sys
from torch.utils.checkpoint import checkpoint


class EvoBlock(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(EvoBlock,self).__init__()
        N_head = 16
        c      = 16
        self.msa_row=EvoMSA.MSARow(m_dim,z_dim,N_head,c)
        self.msa_col=EvoMSA.MSACol(m_dim,N_head,c)
        self.msa_trans=EvoMSA.MSATrans(m_dim)

        self.msa_opm=EvoMSA.MSAOPM(m_dim,z_dim)

        self.pair_triout=EvoPair.TriOut(z_dim,72)
        self.pair_triin =EvoPair.TriIn(z_dim,72)
        self.pair_tristart=EvoPair.TriAttStart(z_dim)
        self.pair_triend  =EvoPair.TriAttEnd(z_dim)
        self.pair_trans = EvoPair.PairTrans(z_dim)
        self.docheck=docheck
        if docheck:
            print('will do checkpoint')

    def layerfunc_msa_row(self,m,z):
        return self.msa_row(m,z) + m
    def layerfunc_msa_col(self,m):
        return self.msa_col(m) + m
    def layerfunc_msa_trans(self,m):
        return self.msa_trans(m) + m
    def layerfunc_msa_opm(self,m,z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self,z):
        return self.pair_triout(z) + z
    def layerfunc_pair_triin(self,z):
        return self.pair_triin(z) + z
    def layerfunc_pair_tristart(self,z):
        return self.pair_tristart(z) + z
    def layerfunc_pair_triend(self,z):
        return self.pair_triend(z) + z      
    def layerfunc_pair_trans(self,z):
        return self.pair_trans(z) + z  
    def forward(self,m,z):
        if True:
            m = m + self.msa_row(m,z)
            m = m + self.msa_col(m)
            m = m + self.msa_trans(m)
            z = z + self.msa_opm(m)
            z = z + self.pair_triout(z)
            z = z + self.pair_triin(z)
            #z = z + self.pair_tristart(z)
            #z = z + self.pair_triend(z)
            z = z + self.pair_trans(z)
            return m,z
        else:
            m=checkpoint(self.layerfunc_msa_row,m,z)
            m=checkpoint(self.layerfunc_msa_col,m)
            m=checkpoint(self.layerfunc_msa_trans,m)
            z=checkpoint(self.layerfunc_msa_opm,m,z)

            z=checkpoint(self.layerfunc_pair_triout,z)
            z=checkpoint(self.layerfunc_pair_triin,z)
            z=checkpoint(self.layerfunc_pair_tristart,z)
            z=checkpoint(self.layerfunc_pair_triend,z)
            z=checkpoint(self.layerfunc_pair_trans,z)

            return m,z


class Evoformer(nn.Module):
    def __init__(self,m_dim,z_dim,N_elayers=12,docheck=False):
        super(Evoformer,self).__init__()
        self.layers=[N_elayers]
        self.docheck=docheck
        if docheck:
            pass
            #print('will do checkpoint')
        self.evos=nn.ModuleList([EvoBlock(m_dim,z_dim,True) for i in range(self.layers[0])])

    def layerfunc(self,layermodule,m,z):
        m_,z_=layermodule(m,z)
        return m_,z_


    def forward(self,m,z):
        
        if True:
            #print('will do checkpoint in Evoformer')
            for i in range(self.layers[0]):
                #m,z=checkpoint(self.layerfunc,self.evos[i],m,z)
                m,z=self.evos[i](m,z)
            return m,z
        else:
            for i in range(self.layers[0]):
                m,z=self.evos[i](m,z)

            return m,z









if __name__ == "__main__":
    N=10
    L=30
    m_dim=16
    z_dim=8
    m=torch.rand(N,L,m_dim)
    z=torch.rand(L,L,z_dim)
    model = Evoformer(m_dim,z_dim)
    m,z=model(m,z)
    print(model.parameters())
    for param in model.parameters():
        print(type(param), param.size())
    print(m.shape,z.shape)