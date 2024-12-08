import math
from numpy import NINF, arccos, arctan2
import torch
from torch import nn
from torch.autograd import Function

def sin_cos_angle(p0,p1,p2):
    # [b 3] 
    b0=p0-p1
    b1=p2-p1

    b0=b0 / (torch.norm(b0,dim=-1,keepdim=True)+1e-08)
    b1=b1 / (torch.norm(b1,dim=-1,keepdim=True)+1e-08)
    recos=torch.sum(b0*b1,-1)
    recos=torch.clamp(recos,-0.9999,0.9999)
    resin = torch.sqrt(1-recos**2)
    return resin,recos


def sin_cos_dihedral(p0,p1,p2,p3):

    #p0 = p[:,0:1,:]
    #p1 = p[:,1:2,:]
    #p2 = p[:,2:3,:]
    #p3 = p[:,3:4,:]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1=b1/(torch.norm(b1,dim=-1,keepdim=True)+1e-8)  
    
    v = b0 - torch.einsum('bj,bj->b', b0, b1)[:,None]*b1
    w = b2 - torch.einsum('bj,bj->b', b2, b1)[:,None]*b1
    x = torch.einsum('bj,bj->b', v, w)
    #print(x)
    y = torch.einsum('bj,bj->b', torch.cross(b1, v,-1), w)
    #print(y.shape)
    torsion_L = torch.norm(torch.cat([x[:,None],y[:,None]],dim=-1),dim=-1)
    x = x / (torsion_L+1e-8)
    y = y / (torsion_L+1e-8)
    return y,x #torch.atan2(y,x)

def dihedral_2d(p0,p1,p2,p3):
    # p : [L,L,3]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2  
    #print(b0.shape)
    b1=b1/(torch.norm(b1,dim=-1,keepdim=True)+1e-8) 
    v = b0 - torch.einsum('abj,abj->ab', b0, b1)[...,None]*b1
    w = b2 - torch.einsum('abj,abj->ab', b2, b1)[...,None]*b1
    x = torch.einsum('abj,abj->ab', v, w)
    y = torch.einsum('abj,abj->ab', torch.cross(b1, v,-1), w)
    return torch.atan2(y,x)
def dihedral_1d(p0,p1,p2,p3):
    # p : [L,L,3]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2  
    print(b0.shape)
    b1=b1/(torch.norm(b1,dim=-1,keepdim=True)+1e-8) 
    v = b0 - torch.einsum('bj,bj->b', b0, b1)[...,None]*b1
    w = b2 - torch.einsum('bj,bj->b', b2, b1)[...,None]*b1
    x = torch.einsum('bj,bj->b', v, w)
    y = torch.einsum('bj,bj->b', torch.cross(b1, v,-1), w)
    return torch.atan2(y,x)

def angle_2d(p0,p1,p2):
    # [a b 3] 
    b0=p0-p1
    b1=p2-p1
    b0=b0 / (torch.norm(b0,dim=-1,keepdim=True)+1e-08)
    b1=b1 / (torch.norm(b1,dim=-1,keepdim=True)+1e-08)
    recos=torch.sum(b0*b1,-1)
    recos=torch.clamp(recos,-0.9999,0.9999)
    return torch.arccos(recos)
def angle_1d(p0,p1,p2):
    return angle_2d(p0,p1,p2)

def distance_2d(p0,p1):
    return  (p0-p1).norm(dim=-1)

def get_omg_map(x):
    # x: L 4 3    N CA C CB
    L=x.shape[0]
    cai=x[:,None,1].repeat(1,L,1)
    cbi=x[:,None,-1].repeat(1,L,1)
    cbj=x[None,:,-1].repeat(L,1,1)
    caj=x[None,:,1].repeat(L,1,1)
    torsion = dihedral_2d(cai,cbi,cbj,caj)
    return torsion

def get_phi_map(x):
    L=x.shape[0]
    cai=x[:,None,1].repeat(1,L,1) 
    cbi=x[:,None,-1].repeat(1,L,1) 
    cbj=x[None,:,-1].repeat(L,1,1)
    return angle_2d(cai,cbi,cbj)

def get_theta_map(x):
    L=x.shape[0]
    ni =x[:,None,0].repeat(1,L,1)
    cai=x[:,None,1].repeat(1,L,1) 
    cbi=x[:,None,-1].repeat(1,L,1) 
    cbj=x[None,:,-1].repeat(L,1,1)
    return dihedral_2d(ni,cai,cbi,cbj)

def get_cadis_map(x):
    cai=x[:,None,1]
    caj=x[None,:,1]
    return distance_2d(cai,caj)

def get_cbdis_map(x):
    cai=x[:,None,-1]
    caj=x[None,:,-1]
    return distance_2d(cai,caj)


def get_all_prot(x):
    L=x.shape[0]
    ni =x[:,None,0].repeat(1,L,1)
    cai=x[:,None,1].repeat(1,L,1)
    ci= x[:,None,2].repeat(1,L,1)
    cbi=x[:,None,-1].repeat(1,L,1)

    nj =x[None,:,0].repeat(L,1,1) 
    caj=x[None,:,1].repeat(L,1,1)  
    cj =x[None,:,2].repeat(L,1,1)    
    cbj=x[None,:,-1].repeat(L,1,1)

    cbmap=distance_2d(cbi,cbj)
    camap=distance_2d(cai,caj)


    omgmap=dihedral_2d(cai,cbi,cbj,caj)
    psimap=angle_2d(cai,cbi,cbj)
    thetamap=dihedral_2d(ni,cai,cbi,cbj)

def get_all(x):
    L = x.shape[0]
    pi= x[:,None,0].repeat(1,L,1)
    ci= x[:,None,1].repeat(1,L,1)
    ni= x[:,None,2].repeat(1,L,1)

    pj= x[None,:,0].repeat(L,1,1)
    cj= x[None,:,1].repeat(L,1,1)
    nj= x[None,:,2].repeat(L,1,1)

    pp = distance_2d(pi,pj)
    cc = distance_2d(ci,cj)
    nn = distance_2d(ni,nj)

    pnn = angle_2d(pi,ni,nj)
    pcc = angle_2d(pi,ci,cj)
    cnn = angle_2d(ci,ni,nj)

    pccp = dihedral_2d(pi,ci,cj,pj)
    pnnp = dihedral_2d(pi,ni,nj,pj)
    cnnc = dihedral_2d(ci,ni,nj,cj)


    return pp,cc,nn,pnn,pcc,cnn,pccp,pnnp,cnnc

