
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
class InvariantPointAttention(nn.Module):
    def __init__(self,dim_in,dim_z,N_head=8,c=16,N_query=4,N_p_values=6,) -> None:
        super(InvariantPointAttention,self).__init__()
        self.dim_in=dim_in
        self.dim_z=dim_z
        self.N_head =N_head
        self.c=c
        self.c_squ = 1.0/math.sqrt(c)
        self.W_c = math.sqrt(2.0/(9*N_query))
        self.W_L = math.sqrt(1.0/3)
        self.N_query=N_query
        self.N_p_values=N_p_values
        self.liner_nb_q1=basic.LinearNoBias(dim_in,self.c*N_head)
        self.liner_nb_k1=basic.LinearNoBias(dim_in,self.c*N_head)
        self.liner_nb_v1=basic.LinearNoBias(dim_in,self.c*N_head)

        self.liner_nb_q2=basic.LinearNoBias(dim_in,N_head*N_query*3)
        self.liner_nb_k2=basic.LinearNoBias(dim_in,N_head*N_query*3)

        self.liner_nb_v3=basic.LinearNoBias(dim_in,N_head*N_p_values*3)

        self.liner_nb_z=basic.LinearNoBias(dim_z,N_head)
        self.lastlinear1=basic.Linear(N_head*dim_z,dim_in)
        self.lastlinear2=basic.Linear(N_head*c,dim_in)
        self.lastlinear3=basic.Linear(N_head*N_p_values*3,dim_in)
        self.gama = nn.ParameterList([nn.Parameter(torch.zeros(N_head))])
        self.cos_f=nn.CosineSimilarity(dim=-1)

    def forward(self,s,z,rot,trans):
        L=s.shape[0]
        q1=self.liner_nb_q1(s).reshape(L,self.N_head,self.c) # Lq, 
        k1=self.liner_nb_k1(s).reshape(L,self.N_head,self.c)
        v1=self.liner_nb_v1(s).reshape(L,self.N_head,self.c) # lv,h,c

        attmap=torch.einsum('ihc,jhc->ijh',q1,k1) * self.c_squ # Lq,Lk_v,h
        bias_z=self.liner_nb_z(z) # L L h

        q2 = self.liner_nb_q2(s).reshape(L,self.N_head,self.N_query,3)
        k2 = self.liner_nb_k2(s).reshape(L,self.N_head,self.N_query,3)

        v3 = self.liner_nb_v3(s).reshape(L,self.N_head,self.N_p_values,3)

        q2 = basic.IPA_transform(q2,rot,trans) # Lq,self.N_head,self.N_query,3
        k2 = basic.IPA_transform(k2,rot,trans) # Lk,self.N_head,self.N_query,3

        dismap=((q2[:,None,:,:,:] - k2[None,:,:,:,:])**2).sum([3,4]) ## Lq,Lk, self.N_head,
        #dismap=dismap - (self.cos_f(q2[:,None,:,:,:] , k2[None,:,:,:,:])).sum(3)
        attmap = attmap + bias_z - F.softplus(self.gama[0])[None,None,:]*dismap*self.W_c*0.5
        #print(torch.max(attmap*self.W_L),torch.min(attmap)*self.W_L)
        #attmap = F.softmax( torch.clamp(attmap*self.W_L,-5,5),dim=1   ) # Lk dim,  [Lq,Lk, self.N_head]

        attmap = F.softmax( attmap*self.W_L,dim=1   ) # Lk dim,  [Lq,Lk, self.N_head]
        if attdrop:
            if self.training:
                attmap = basic.DropAtt(attmap,dim=1)
        o1 = (attmap[:,:,:,None] * z[:,:,None,:]).sum(1) # Lq, N_head, c_z
        o2 = torch.einsum('abc,dab->dbc',v1,attmap) # Lq, N_head, c
        o3 = basic.IPA_transform(v3,rot,trans) # Lv, h, p* ,3
        o3 = basic.IPA_inverse_transform( torch.einsum('vhpt,gvh->ghpt',o3,attmap),rot,trans) #Lv, h, p* ,3

        return self.lastlinear1(o1.reshape(L,-1)) + self.lastlinear2(o2.reshape(L,-1)) + self.lastlinear3(o3.reshape(L,-1)) 



        


if __name__ == "__main__":
    dim_in,dim_z = 8,4
    L = 10
    ipa = InvariantPointAttention(dim_in,dim_z)
    s=torch.rand(L,dim_in)
    z=torch.rand(L,L,dim_z)
    rot=(torch.eye(3)[None,:,:]).repeat(L,1,1)
    trans=torch.rand(L,3)

    out=ipa(s,z,rot,trans)
    print(out)
    print(out.shape)






