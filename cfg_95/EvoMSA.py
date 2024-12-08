from numpy import select
import torch
from torch import nn
from torch.nn import functional as F
import basic
import math
import os


expdir=os.path.dirname(os.path.abspath(__file__))
lines = open(os.path.join(expdir,'newconfig')).readlines()
attdrop = lines[0].strip().split()[-1] == '1'
denoisee2e = lines[1].strip().split()[-1] == '1'
ss_type =  lines[2].strip().split()[-1] 
    

    





def SignedSqrt( x):
    x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
    return x
class MSARow(nn.Module):
    def __init__(self,m_dim,z_dim,N_head=8,c=8):
        super(MSARow,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(m_dim)
        self.qlinear = basic.LinearNoBias(m_dim,N_head*c)
        self.klinear = basic.LinearNoBias(m_dim,N_head*c)
        self.vlinear = basic.LinearNoBias(m_dim,N_head*c)
        self.norm_z  = nn.LayerNorm(z_dim)
        self.zlinear = basic.LinearNoBias(z_dim,N_head)
        self.glinear = basic.Linear(m_dim,N_head*c)
        self.olinear = basic.Linear(N_head*c,m_dim)

    def forward(self,m,z):
        # m : N L 32  
        N,L,D = m.shape  
        m = self.norm1(m)
        q = self.qlinear(m).reshape(N,L,self.N_head,self.c) #s rq h c 
        k = self.klinear(m).reshape(N,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(m).reshape(N,L,self.N_head,self.c)
        b = self.zlinear(self.norm_z(z))
        g = torch.sigmoid(self.glinear(m)).reshape(N,L,self.N_head,self.c)
        att=torch.einsum('bqhc,bvhc->bqvh',q,k) * (self.sq_c) + b[None,:,:,:] # rq rv h
        att=F.softmax(SignedSqrt(att),dim=2)
        if attdrop:
            if self.training:
                att = basic.DropAtt(att,dim=2)
        o = torch.einsum('bqvh,bvhc->bqhc',att,v) * g
        m_ = self.olinear(o.reshape(N,L,-1))
        return m_

class MSACol(nn.Module):
    def __init__(self,m_dim,N_head=8,c=8):
        super(MSACol,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(m_dim)
        self.qlinear = basic.LinearNoBias(m_dim,N_head*c)
        self.klinear = basic.LinearNoBias(m_dim,N_head*c)
        self.vlinear = basic.LinearNoBias(m_dim,N_head*c)

        self.glinear = basic.Linear(m_dim,N_head*c)
        self.olinear = basic.Linear(N_head*c,m_dim)

    def forward(self,m):
        # m : N L 32  
        N,L,D = m.shape  
        m = self.norm1(m)
        q = self.qlinear(m).reshape(N,L,self.N_head,self.c) #s rq h c 
        k = self.klinear(m).reshape(N,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(m).reshape(N,L,self.N_head,self.c)

        g = torch.sigmoid(self.glinear(m)).reshape(N,L,self.N_head,self.c)

        att=torch.einsum('slhc,tlhc->stlh',q,k) * (self.sq_c)  # rq rv h
        att=F.softmax(SignedSqrt(att),dim=1)
        if attdrop:
            if self.training:
                att = basic.DropAtt(att,dim=1)
        o = torch.einsum('stlh,tlhc->slhc',att,v) * g
        m_ = self.olinear(o.reshape(N,L,-1))
        return m_

class MSATrans(nn.Module):
    def __init__(self,m_dim,c_expand=2):
        super(MSATrans,self).__init__()
        self.c_expand=4
        self.m_dim=m_dim
        self.norm=nn.LayerNorm(m_dim)
        self.linear1 = basic.Linear(m_dim,m_dim*c_expand)
        self.linear2 = basic.Linear(m_dim*c_expand,m_dim)
    def forward(self,m):
        m = self.norm(m)
        m = self.linear1(m)
        m = self.linear2(F.relu(m))
        return m

class MSAOPM(nn.Module):
    def __init__(self,m_dim,z_dim,c=12):
        super(MSAOPM,self).__init__()
        self.m_dim=m_dim
        self.c=c
        self.norm=nn.LayerNorm(m_dim)
        self.linear1=basic.Linear(m_dim,c)
        self.linear2=basic.Linear(m_dim,c)
        self.linear3=basic.Linear(c*c,z_dim)
    def forward(self,m):
        N,L,D=m.shape
        o=self.norm(m)
        a=self.linear2(o)
        b=self.linear1(o)
        o = torch.einsum('nia,njb->nijab',a,b).mean(dim=0)
        o = self.linear3(o.reshape(L,L,-1))
        return o


        



if __name__ == "__main__":
    N=10
    L=30
    m_dim=16
    z_dim=8
    m=torch.rand(N,L,m_dim)
    z=torch.rand(L,L,z_dim)
    msarow=MSARow(m_dim,z_dim)
    msacol=MSACol(m_dim)
    msatrans=MSATrans(m_dim)
    msaopm=MSAOPM(m_dim,z_dim)
    y=msaopm(m)
    print(y.shape)