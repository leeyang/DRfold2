import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np 
import math,sys,math
from subprocess import Popen, PIPE, STDOUT
from io import BytesIO
import os
from torch.autograd import Function

def coor_selection(coor,mask):
    #[L,n,3],[L,n],byte
    return torch.masked_select(coor,mask.bool()).view(-1,3)


def pair_distance(x1,x2,eps=1e-6,p=2):
    n1=x1.size()[0]
    n2=x2.size()[0]
    x1_=x1.view(n1,1,-1)
    x2_=x2.view(1,n2,-1)
    x1_=x1_.expand(n1,n2,-1)
    x2_=x2_.expand(n1,n2,-1)
    diff = torch.abs(x1_ - x2_)
    out = torch.pow(diff + eps, p).sum(dim=2)
    return torch.pow(out, 1. / p)  

class torsion(Function):
    #PyTorch class to calculate differentiable torsion angle
    #https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    #https://salilab.org/modeller/manual/node492.html
    @staticmethod
    def forward_(ctx,input1,input2,input3,input4):
        #Lx3
        # 0       1     2       3
        inputi,inputj,inputk,inputl=input1.detach().numpy(),input2.detach().numpy(),input3.detach().numpy(),input4.detach().numpy()
        rij,rkj,rkl=inputi-inputj,inputk-inputj,inputk-inputl
        corss_ijkj=np.cross(rij,rkj)
        cross_kjkl=np.cross(rkj,rkl)
        angle=np.sum(corss_ijkj*cross_kjkl,axis=-1)
        angle=angle/(np.linalg.norm(corss_ijkj,axis=-1)*np.linalg.norm(cross_kjkl,axis=-1))
        signlamda=np.sign(np.sum(rkj*np.cross(corss_ijkj,cross_kjkl),-1))
        angle[angle<-1]=-1
        angle[angle>1]=1
        ctx.mj=corss_ijkj
        ctx.nk=cross_kjkl
        ctx.rkj=rkj
        ctx.rij=rij
        ctx.rkl=rkl
        #ctx.save_for_backward(input1,input2,input3,input4)
        return torch.as_tensor(np.arccos(angle)*signlamda,dtype=input1.dtype)
    @staticmethod
    def forward(ctx,input1,input2,input3,input4):
        #Lx3
        p0,p1,p2,p3=input1.detach().numpy(),input2.detach().numpy(),input3.detach().numpy(),input4.detach().numpy()
        b0_ = -(p1-p0)
        b1_ = p2-p1
        b2_ = p3-p2
        ctx.rkj=b1_+0.0
        ctx.rij=b0_+0.0
        ctx.rkl=-b2_  +0.0     
        ctx.mj=np.cross(ctx.rij,ctx.rkj)
        ctx.nk=np.cross(ctx.rkj,ctx.rkl)
        b1 =b1_ / np.linalg.norm(b1_,axis=-1,keepdims=True)
        v = b0_ -  (b0_*b1).sum(-1,keepdims=True)   *b1
        w = b2_ - (b2_*b1).sum(-1,keepdims=True)*b1
        x = (v*w).sum(-1)
        y = (np.cross(b1, v)*w).sum(-1)
        #print(x.shape)

        return torch.as_tensor(np.arctan2(y, x),dtype=input1.dtype)
    @staticmethod
    def backward(ctx,grad_output):
        rij,rkj,rkl=ctx.rij,ctx.rkj,ctx.rkl
        rnk,rmj=ctx.nk,ctx.mj
        dkj=np.linalg.norm(rkj,axis=-1,keepdims=True)
        dmj=np.linalg.norm(rmj,axis=-1,keepdims=True)
        dnk=np.linalg.norm(rnk,axis=-1,keepdims=True)
        grad1=(dkj/((dmj*dmj)))*rmj
        grad4=-(dkj/((dnk*dnk)))*rnk
        
        grad2=( (rij*rkj).sum(-1,keepdims=True)/((dkj*dkj))-1    )*grad1 - (rkl*rkj).sum(-1,keepdims=True)/((dkj*dkj))*grad4
        grad3=( (rkl*rkj).sum(-1,keepdims=True)/((dkj*dkj))-1    )*grad4 - (rij*rkj).sum(-1,keepdims=True)/((dkj*dkj))*grad1
        
        grad1,grad2,grad3,grad4=torch.from_numpy(grad1),torch.from_numpy(grad2),torch.from_numpy(grad3),torch.from_numpy(grad4)
        return grad1*grad_output[:,None],grad2*grad_output[:,None],grad3*grad_output[:,None],grad4*grad_output[:,None]


def dihedral(input1,input2,input3,input4):
    return torsion.apply(input1,input2,input3,input4)



def angle(p0,p1,p2):
    # [b 3] 
    b0=p0-p1
    b1=p2-p1

    b0=b0 / (torch.norm(b0,dim=-1,keepdim=True)+1e-08)
    b1=b1 / (torch.norm(b1,dim=-1,keepdim=True)+1e-08)
    recos=torch.sum(b0*b1,-1)
    recos=torch.clamp(recos,-0.9999,0.9999)
    #print(recos.shape)
    return torch.acos(recos)


def rigidFrom3Points(x):
    x1,x2,x3 = x[:,0],x[:,1],x[:,2]
    v1=x3-x2
    v2=x1-x2
    e1=v1/(torch.norm(v1,dim=-1,keepdim=True) + 1e-06)
    u2=v2 - e1*(torch.einsum('bn,bn->b',e1,v2)[:,None])
    e2 = u2/(torch.norm(u2,dim=-1,keepdim=True) + 1e-06)
    e3=torch.cross(e1,e2,dim=-1)

    return torch.stack([e1,e2,e3],dim=1)



def Kabsch_rigid(bases,x1,x2,x3):
    '''
    return the direction from to_q to from_p
    '''
    the_dim=1
    to_q = torch.stack([x1,x2,x3],dim=the_dim)
    biasq=torch.mean(to_q,dim=the_dim,keepdim=True)
    q=to_q-biasq
    m = torch.einsum('bnz,bny->bzy',bases,q)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r,biasq.squeeze()


def Get_base(seq,basenpy_standard):
    base_num = basenpy_standard.shape[1]
    basenpy = np.zeros([len(seq),base_num,3])
    seqnpy = np.array(list(seq))
    basenpy[seqnpy=='A']=basenpy_standard[0]
    basenpy[seqnpy=='a']=basenpy_standard[0]

    basenpy[seqnpy=='G']=basenpy_standard[1]
    basenpy[seqnpy=='g']=basenpy_standard[1]

    basenpy[seqnpy=='C']=basenpy_standard[2]
    basenpy[seqnpy=='c']=basenpy_standard[2]

    basenpy[seqnpy=='U']=basenpy_standard[3]
    basenpy[seqnpy=='u']=basenpy_standard[3]

    basenpy[seqnpy=='T']=basenpy_standard[3]
    basenpy[seqnpy=='t']=basenpy_standard[3]
    return torch.from_numpy(basenpy).double()