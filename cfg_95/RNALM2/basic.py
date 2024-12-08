import math
import torch
from torch import nn
import random
class Linear(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(Linear,self).__init__()
        self.linear = nn.Linear(dim_in,dim_out)
    def forward(self,x):
        x = self.linear(x)
        return x


class LinearNoBias(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(LinearNoBias,self).__init__()
        self.linear = nn.Linear(dim_in,dim_out,bias=False)
    def forward(self,x):
        x = self.linear(x)
        return x
    


def transform(k,rotation,translation):
    # K L x 3
    # rotation
    return torch.matmul(k,rotation) + translation


def batch_transform(k,rotation,translation):
    # k:            L 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('ba,bad->bd',k,rotation) + translation

def batch_atom_transform(k,rotation,translation):
    # k:            L N 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('bja,bad->bjd',k,rotation) + translation[:,None,:]

def IPA_transform(k,rotation,translation):
    # k:            L d1, d2, 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('bija,bad->bijd',k,rotation)+translation[:,None,None,:]

def IPA_inverse_transform(k,rotation,translation):
    # k:            L d1, d2, 3
    # rotation:     L 3 x 3
    # translation:  L 3   
    return torch.einsum('bija,bad->bijd',k-translation[:,None,None,:],rotation.transpose(-1,-2))

def update_transform(t,tr,rotation,translation):
    return torch.einsum('bja,bad->bjd',t,rotation),torch.einsum('ba,bad->bd',tr,rotation) +translation


def quat2rot(q,L):
    scale= ((q**2).sum(dim=-1,keepdim=True)    +1) [:,:,None]
    u=torch.empty([L,3,3],device=q.device)
    u[:,0,0]=1*1+q[:,0]*q[:,0]-q[:,1]*q[:,1]-q[:,2]*q[:,2]
    u[:,0,1]=2*(q[:,0]*q[:,1]-1*q[:,2])
    u[:,0,2]=2*(q[:,0]*q[:,2]+1*q[:,1])
    u[:,1,0]=2*(q[:,0]*q[:,1]+1*q[:,2])
    u[:,1,1]=1*1-q[:,0]*q[:,0]+q[:,1]*q[:,1]-q[:,2]*q[:,2]
    u[:,1,2]=2*(q[:,1]*q[:,2]-1*q[:,0])
    u[:,2,0]=2*(q[:,0]*q[:,2]-1*q[:,1])
    u[:,2,1]=2*(q[:,1]*q[:,2]+1*q[:,0])
    u[:,2,2]=1*1-q[:,0]*q[:,0]-q[:,1]*q[:,1]+q[:,2]*q[:,2]
    return u/scale


def rotation_x(sintheta,costheta,ones,zeros):
    # L x 1
    return torch.stack([torch.stack([ones, zeros, zeros]),
                     torch.stack([zeros, costheta, sintheta]),
                     torch.stack([zeros, -sintheta, costheta])])
def rotation_y(sintheta,costheta,ones,zeros):
    # L x 1
    return torch.stack([torch.stack([costheta, zeros, sintheta]),
                     torch.stack([zeros, ones, zeros]),
                     torch.stack([-sintheta, zeros, costheta])])
def rotation_z(sintheta,costheta,ones,zeros):
    # L x 1
    return torch.stack([torch.stack([costheta, sintheta, zeros]),
                     torch.stack([-sintheta, costheta, zeros]),
                     torch.stack([zeros, zeros, ones])])
def batch_rotation(k,rotation):
    # k:            L 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('ba,bad->bd',k,rotation) 

def compute_cb(bl,sin_angle,cos_angle,sin_torsion,cos_torsion):
    L=bl.shape[0]
    ones=torch.ones(L,device=bl.device)
    zeros=torch.zeros(L,device=bl.device)
    cb=torch.stack([bl,zeros,zeros]).permute(1,0)
    rotz=rotation_z(sin_angle,cos_angle,ones,zeros).permute(2,0,1)
    rotx=rotation_x(sin_torsion,cos_torsion,ones,zeros).permute(2,0,1)
    cb=batch_rotation(cb,rotz)
    cb=batch_rotation(cb,rotx)
    return cb

def rigidFrom3Points_(x1,x2,x3):
    v1=x3-x2
    v2=x1-x2
    e1=v1/(torch.norm(v1,dim=-1,keepdim=True) + 1e-03)
    u2=v2 - e1*(torch.einsum('bn,bn->b',e1,v2)[:,None])
    e2 = u2/(torch.norm(u2,dim=-1,keepdim=True) + 1e-03)
    e3=torch.cross(e1,e2,dim=-1)

    return torch.stack([e1,e2,e3],dim=1),x2[:,:]
def rigidFrom3Points(x1,x2,x3):# L 3
    the_dim=1
    x = torch.stack([x1,x2,x3],dim=the_dim)
    x_mean = torch.mean(x,dim=the_dim,keepdim=True)
    x = x - x_mean


    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r,x_mean.squeeze()
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
def Generate_msa_mask(n,l):
    # 1: 15% mask out
    randommatrix=torch.rand(n,l)
    mask = (randommatrix <0.1).float()
    # 2 random a segment
    seqlength = int(l*0.1)
    sindex=round(random.random()*(l-seqlength))
    endindex=min(l,sindex+seqlength)
    mask[:,sindex:endindex]=1
    return mask


