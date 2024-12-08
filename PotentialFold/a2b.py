import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.optim as opt
def _base_frame():
    x1=torch.FloatTensor([-4.2145e-01,  3.7763e+00,0])[None,:]
    x2=torch.FloatTensor([0,0,0])[None,:]
    x3=torch.FloatTensor([3.3910e+00,0,0])[None,:]
    #x4=torch.FloatTensor([-5.2283e-01,-7.7104e-01,-1.2162e+00])[None,:]
    x=torch.cat([x1,x2,x3])
    return x.double()
BASEFRAME = _base_frame()
def getmatrix(alpha,beta_,d,L):
    ze=torch.zeros(L,dtype=torch.float64)
    ons=torch.ones(L,dtype=torch.float64)
    beta=ons*beta_
    ma1=torch.stack([torch.cos(beta),torch.sin(alpha)*torch.sin(beta),torch.cos(alpha)*torch.sin(beta),d*torch.cos(beta)])
    ma2=torch.stack([ze,torch.cos(alpha),-torch.sin(alpha),ze])
    ma3=torch.stack([-torch.sin(beta),torch.sin(alpha)*torch.cos(beta),torch.cos(alpha)*torch.cos(beta),-d*torch.sin(beta)])
    ma4=torch.stack([ze,ze,ze,ons])
    ma=torch.stack([ma1,ma2,ma3,ma4])
    return ma


def getallmatrix(angles,number,bm4):
    ##### The first elements are useless (some)
    bond_angle=1.65
    bond_angle2=1.81
    bond_length2=3.91
    bond_length=3.87
    bm2=getmatrix(angles[:,0],torch.DoubleTensor([np.pi-0.1]+[np.pi-bond_angle]*(number-1)),torch.DoubleTensor([bond_length]*number),number)
    bm3=getmatrix(angles[:,1],torch.DoubleTensor([np.pi-bond_angle2]*number),torch.DoubleTensor([bond_length2]*number),number) # 1.9391

    Bm=torch.stack([bm2,bm3])

    Bm=Bm.permute(3,0,1,2).contiguous().view(-1,4,4)
    Amatrix=[]
    Amatrix.append(Bm[0]*1.)
    for i in range(len(Bm)-1):
        Amatrix.append(torch.mm(Amatrix[i],Bm[i+1]))
    Amatrix=torch.stack(Amatrix)[:,:-1]
    Bmatrix=Amatrix.contiguous().view(-1,2,3,4)[:,1] # CA_matrix   
    return Amatrix,Bmatrix

def batch_nerf(a, b, c, l, theta, chi):
    #l, theta, chi=torch.DoubleTensor([l, theta, chi])[:]
    W_hat = torch.nn.functional.normalize(b - a, dim=-1)
    x_hat = torch.nn.functional.normalize(c-b, dim=-1)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=-1)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=-1)

    # calculate coord pre rotation matrix
    d = torch.stack([torch.squeeze(-l * torch.cos(theta)),
                     torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
                     torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])

    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    d = d.transpose(0,1).unsqueeze(2)
    #print(d.shape,M.shape)
    #res = c + torch.mm(M, d).squeeze()
    res = c + torch.einsum('bnm,bmd->bnd', M,d).squeeze()
    #print(d.shape,M.shape,res.shape)
    return res


def a2b(angles):
    # angles[L*3] N-CA-C-N...
    # beta c-p-c-n
    number=angles.shape[0]
    beta=angles[:,-1]
    pcn=torch.DoubleTensor([1.63]*number)
    cn=torch.DoubleTensor([3.36]*number)
    init=torch.DoubleTensor([0,0,0,1])
    Amatrix,Bmatrix=getallmatrix(angles[:,:2],number,beta)
    coor=torch.matmul(Amatrix,init)
    coor=coor.view(-1,2,3)
    coor=torch.cat([coor[[0]]*0,coor],dim=0)
    ns=batch_nerf(coor[:-1,1], coor[1:,0], coor[1:,1], cn, pcn, beta)[:,None,:]
    
    return torch.cat([coor[1:],ns],dim=1)
def batch_atom_transform(k,rotation,translation):
    # k:            L N 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('bja,bad->bjd',k,rotation) + translation[:,None,:]
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
def rigidFrom3Points(x1,x2,x3):
    v1=x3-x2
    v2=x1-x2
    e1=v1/(torch.norm(v1,dim=-1,keepdim=True) + 1e-03)
    u2=v2 - e1*(torch.einsum('bn,bn->b',e1,v2)[:,None])
    e2 = u2/(torch.norm(u2,dim=-1,keepdim=True) + 1e-08)
    e3=torch.cross(e1,e2,dim=-1)

    return torch.stack([e1,e2,e3],dim=1),x2[:,:]
def Non2rot(x,L):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r

def quat2b(baseframe,x):
    #x: L 6
    L = x.shape[0]
    #predx=( baseframe.repeat(L,1,1) )
    #rot = quat2rot(x[:,:3],L)
    rot = Non2rot(x[:,:9],L)
    
    trans=x[:,9:]
    predx = batch_atom_transform(baseframe,rot,trans)
    return predx
    



if __name__ == '__main__': 
    import geo
    L=10
    angles=torch.rand(L,3)
    coor=a2b(angles)
    print(coor.shape)
    for i in range(L-1):
        # pcp=geo.angle_1d(coor[i,0],coor[i,1],coor[i+1,0])
        # cpc=geo.angle_1d(coor[i+0,1],coor[i+1,0],coor[i+1,1])
        # pc=geo.distance_2d(coor[i+0,0],coor[i+0,1])
        # cp=geo.distance_2d(coor[i+0,1],coor[i+1,0])
        # print(pcp)
        # print(cpc)
        # print(pc)
        # print(cp)
        pcn=geo.angle_1d(coor[i,0],coor[i,1],coor[i,2])
        cn=geo.distance_2d(coor[i+0,1],coor[i+0,2])
        
        print(pcn)
        print(cn)
    cpcn=geo.dihedral_1d(coor[:-1,1],coor[1:,0],coor[1:,1],coor[1:,2])
    print(cpcn-angles[1:,-1])