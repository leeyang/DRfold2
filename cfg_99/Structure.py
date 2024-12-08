import torch
from torch import nn
from torch.nn import functional as F
import basic,IPA
import math,os
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
expdir=os.path.dirname(os.path.abspath(__file__))
lines = open(os.path.join(expdir,'newconfig')).readlines()
attdrop = lines[0].strip().split()[-1] == '1'
denoisee2e = lines[1].strip().split()[-1] == '1'
ss_type =  lines[2].strip().split()[-1] 

class TransitionModule(nn.Module):
    def __init__(self,c):
        super(TransitionModule,self).__init__()
        self.c=c
        self.norm1=nn.LayerNorm(c)
        self.linear1=basic.Linear(c,c)
        self.linear2=basic.Linear(c,c)
        self.linear3=basic.Linear(c,c)
        self.norm2=nn.LayerNorm(c)
    def forward(self,s_):
        s = self.norm1(s_)
        s = F.relu(   self.linear1(s) )
        s = F.relu(   self.linear2(s) )
        s = s_ + self.linear3(s)
        return self.norm2(s)

class BackboneUpdate(nn.Module):
    def __init__(self,indim):
        super(BackboneUpdate,self).__init__()
        self.indim=indim
        self.linear=basic.Linear(indim,6)
        torch.nn.init.zeros_(self.linear.linear.weight)
        torch.nn.init.zeros_(self.linear.linear.bias)
    def forward(self,s,L):
        pred=self.linear(s)
        rot=basic.quat2rot(pred[...,:3],L)
        return rot,pred[...,3:] #rot, translation

class TorsionNet(nn.Module):
    def __init__(self,s_dim,c):
        super(TorsionNet,self).__init__()
        self.s_dim=s_dim
        self.c=c
        self.linear1=basic.Linear(s_dim,c)
        self.linear2=basic.Linear(c,c)

        self.linear3=basic.Linear(c,c)
        self.linear4=basic.Linear(c,c) 

        self.linear5=basic.Linear(c,c)
        self.linear6=basic.Linear(c,c)

        self.linear7_1=basic.Linear(c,1)
        self.linear7_2=basic.Linear(c,2)
        self.linear7_3=basic.Linear(c,2)
    
    def forward(self,s_init,s):
        a = self.linear1(s_init) + self.linear2(s)
        a = a + self.linear4(F.relu(self.linear3(F.relu(a))))
        a = a + self.linear6(F.relu(self.linear5(F.relu(a))))
        bondlength = self.linear7_1(F.relu(a))
        angle = self.linear7_2(F.relu(a))
        torsion = self.linear7_3(F.relu(a))

        angle_L=torch.norm(angle,dim=-1,keepdim=True)
        angle = angle / (angle_L+1e-8)

        torsion_L = torch.norm(torsion,dim=-1,keepdim=True)
        torsion = torsion / (torsion_L+1e-8)

        return bondlength,angle,angle_L,torsion,torsion_L


class lddtpredictor(nn.Module):
    def __init__(self,s_dim,z_dim):
        super(lddtpredictor,self).__init__()

        self.lineara=basic.Linear(s_dim,z_dim)
        self.linearb=basic.Linear(s_dim,z_dim)
        self.linear2=basic.Linear(z_dim,z_dim)
        self.linear3=basic.Linear(z_dim,z_dim)
        self.linear4=basic.Linear(z_dim,z_dim) 
        self.linear5=basic.Linear(3,z_dim)
        self.linear6=basic.Linear(z_dim,z_dim)
        self.linear7=basic.Linear(z_dim,z_dim)
        self.linear8=basic.Linear(z_dim,z_dim)
        self.lastlinear = basic.Linear(z_dim,5) # <1,<2,<4,<8,>4

    def forward(self,rot,trans,s,z):
        zs = self.lineara(s)[None,:,:] + self.linearb(s)[:,None,:]
        zs = zs + self.linear4(F.relu(self.linear3(F.relu(zs))  + self.linear2(z)  ))
        pred_x = trans[:,None,None,:] - trans[None,:,None,:] # Lx Lrot N , 3
        pred_x = torch.einsum('ijnd,jde->ijne',pred_x,rot.transpose(-1,-2))[:,:,0,:] # transpose should be equal to inverse
        zs = zs+ self.linear6(F.relu(self.linear5(F.relu(pred_x))))
        zs = zs+ self.linear8(F.relu(self.linear7(F.relu(zs))))
        return F.log_softmax(self.lastlinear(zs),dim=-1)
    
    def compute(self,predlogsoft):
        explddt = torch.exp(predlogsoft)
        lddt = (explddt[...,:1].sum(dim=-1) + explddt[...,:2].sum(dim=-1) + explddt[...,:3].sum(dim=-1) + explddt[...,:4].sum(dim=-1))*0.25
        return lddt


    

class StructureModule(nn.Module):
    def __init__(self,s_dim,z_dim,N_layer,c):
        super(StructureModule,self).__init__()
        self.s_dim=s_dim
        self.z_dim=z_dim
        self.N_layer=N_layer
        self.N_head=8
        self.c=c
        self.use_rmsdloss=False
        self.layernorm_s=nn.LayerNorm(s_dim)
        self.layernorm_z=nn.LayerNorm(z_dim)
        self.baseframe=self._base_frame()
        # shared weights part
        self.ipa=IPA.InvariantPointAttention(c,z_dim,c)
        self.transition = TransitionModule(c)
        self.bbupdate = BackboneUpdate(c)
        self.torsionnet=TorsionNet(s_dim,c)
        self.lddt_preder= lddtpredictor(s_dim,z_dim)
        self._init_T()
    def _base_frame(self):
        x1=torch.FloatTensor([-4.2145e-01,  3.7763e+00,0])[None,:]
        x2=torch.FloatTensor([0,0,0])[None,:]
        x3=torch.FloatTensor([3.3910e+00,0,0])[None,:]
        #x4=torch.FloatTensor([-5.2283e-01,-7.7104e-01,-1.2162e+00])[None,:]
        x=torch.cat([x1,x2,x3])
        return x
    def _init_T(self):
        self.trans = torch.zeros(3)[None,:]
        self.rot = torch.eye(3)[None,:,:]

    def randomT(self,L,sigmaR=0.1,sigmat=0.1):
        randTrans = np.random.normal(0, sigmat, (L,3))
        randRotation = Rotation.random(L).as_matrix()
        identitym = np.eye(3)
        randRotation = [Rotation.from_matrix(np.stack([randRotation[i],identitym])).mean(weights=[sigmaR,1]).as_matrix() for i in range(L)]
        randRotation = np.array(randRotation)
        return randRotation,randTrans

    



    def pred(self,s_init,z,base_x):
        if self.trans.device != s_init.device:
            self.trans=self.trans.to(s_init.device)
        if self.rot.device != s_init.device:
            self.rot=self.rot.to(s_init.device)
        L=s_init.shape[0]
        rot,trans=self.rot.repeat(L,1,1),self.trans.repeat(L,1)
        # if denoisee2e:
        #     rot,trans = self.randomT(L)
        #     rot,trans = torch.FloatTensor(rot).to(s_init.device),torch.FloatTensor(trans).to(s_init.device)

        s = self.layernorm_s(s_init)
        z = self.layernorm_z(z)
        for layer in range(self.N_layer):
            s = s+ self.ipa(s,z,rot,trans)
            s = self.transition(s)
            rot_tmp,trans_tmp = self.bbupdate(s,L)
            rot,trans = basic.update_transform(rot_tmp,trans_tmp,rot,trans)
            
        s = s+ self.ipa(s,z,rot,trans)
        s = self.transition(s) 
        rot_tmp,trans_tmp = self.bbupdate(s,L)
        rot,trans = basic.update_transform(rot_tmp,trans_tmp,rot,trans)
        #bondlength,angle,angle_L,torsion,torsion_L = self.torsionnet(s_init,s)
        # if self.baseframe.device != s_init.device:
        #     self.baseframe=self.baseframe.to(s_init.device)
        predx=base_x + 0.0
        predlddt = self.lddt_preder(rot,trans,s,z)
        #cb = basic.compute_cb(bondlength[...,0],angle[...,0],angle[...,1],torsion[...,0],torsion[...,1])[:,None,:] # L 3
        #predx = torch.cat([predx,cb],dim=1)
        predx = basic.batch_atom_transform(predx,rot,trans)
        
        return predx,rot,trans,self.lddt_preder.compute(predlddt)








    
        







        

