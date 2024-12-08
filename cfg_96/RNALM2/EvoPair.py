import torch
from torch import nn
from torch.nn import functional as F
from RNALM2 import basic
import math

def SignedSqrt( x):
    x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
    return x
class TriOut(nn.Module):
    def __init__(self,z_dim,c=32):
        super(TriOut,self).__init__()
        self.z_dim = z_dim
        self.norm =nn.LayerNorm(z_dim)
        self.onorm =nn.LayerNorm(c)
        self.alinear=basic.Linear(z_dim,c)
        self.blinear=basic.Linear(z_dim,c)
        self.aglinear=basic.Linear(z_dim,c)
        self.bglinear=basic.Linear(z_dim,c)
        self.glinear =basic.Linear(z_dim,z_dim)
        self.olinear=basic.Linear(c,z_dim)

    def forward(self,z):
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.blinear(z) * torch.sigmoid(self.bglinear(z))
        o = torch.einsum('ilc,jlc->ijc',a,b)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o

class TriIn(nn.Module):
    def __init__(self,z_dim,c=32):
        super(TriIn,self).__init__()
        self.z_dim = z_dim
        self.norm =nn.LayerNorm(z_dim)
        self.onorm =nn.LayerNorm(c)
        self.alinear=basic.Linear(z_dim,c)
        self.blinear=basic.Linear(z_dim,c)
        self.aglinear=basic.Linear(z_dim,c)
        self.bglinear=basic.Linear(z_dim,c)
        self.glinear =basic.Linear(z_dim,z_dim)
        self.olinear=basic.Linear(c,z_dim)

    def forward(self,z):
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.blinear(z) * torch.sigmoid(self.bglinear(z))
        o = torch.einsum('lic,ljc->ijc',a,b)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o


class TriAttStart(nn.Module):
    def __init__(self,z_dim,N_head=4,c=8):
        super(TriAttStart,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)

    def forward(self,z_):
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('blhc,bkhc->blkh',q,k)*self.sq_c + b[None,:,:,:]
        att = F.softmax(SignedSqrt(att),dim=2)
        o = torch.einsum('blkh,bkhc->blhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        return o

class TriAttEnd(nn.Module):
    def __init__(self,z_dim,N_head=4,c=8):
        super(TriAttEnd,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)

    def forward(self,z_):
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('blhc,kbhc->blkh',q,k)*self.sq_c + b[None,:,:,:].permute(0,2,1,3)
        att = F.softmax(SignedSqrt(att),dim=2)
        o = torch.einsum('blkh,klhc->blhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        return o
    def forward2(self,z_):
        z = z_.permute(1,0,2)
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('blhc,bkhc->blkh',q,k)*self.sq_c + b[None,:,:,:]
        att = F.softmax(att,dim=2)
        o = torch.einsum('blkh,bkhc->blhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        o = o.permute(1,0,2)
        return o
class PairTrans(nn.Module):
    def __init__(self,z_dim,c_expand=2):
        super(PairTrans,self).__init__()
        self.z_dim=z_dim
        self.c_expand=c_expand
        self.norm = nn.LayerNorm(z_dim)
        self.linear1=basic.Linear(z_dim,z_dim*c_expand)
        self.linear2=basic.Linear(z_dim*c_expand,z_dim)
    def forward(self,z):
        a = self.linear1(self.norm(z))
        a = self.linear2(F.relu(a))
        return a 




if __name__ == "__main__":
    N=10
    L=30
    m_dim=16
    z_dim=8
    m=torch.rand(N,L,m_dim)
    z=torch.rand(L,L,z_dim)

    tr1=TriAttEnd(z_dim)
    tr2=PairTrans(z_dim)
    y=tr1(z)
    y2=tr1.forward2(z)
    y3=tr2(z)
    print(y3.shape)

