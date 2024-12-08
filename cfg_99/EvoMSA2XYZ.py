from numpy import select
import torch
from torch import nn
from torch.nn import functional as F

import basic,Structure,Evoformer,EvoPair,EvoMSA
import math,sys,os
from torch.utils.checkpoint import checkpoint
import numpy as np
import EvoPair



device = sys.argv[1]
expdir=os.path.dirname(os.path.abspath(__file__))

# from pathlib import Path
# path = Path(expdir)
# parepath = path.parent.absolute()
# sys.path.append(parepath)
# print(parepath)

from RNALM2 import Model


lmcfg={}
lmcfg['s_in_dim']=5
lmcfg['z_in_dim']=2
lmcfg['s_dim']= 512
lmcfg['z_dim']= 128
lmcfg['N_elayers']=18
RNAlm = Model.RNA2nd(lmcfg)


saved_model =  os.path.join(  os.path.dirname(expdir), 'model_hub', 'RCLM','epoch_67000')
RNAlm.load_state_dict(torch.load(saved_model,map_location=torch.device('cpu')),strict=False)
RNAlm.to(device)
RNAlm.eval()
lmaadic = {
            'A':0,'G':1,'C':2,'U':3,'a':0,'g':1,'c':2,'u':3,'T':3,'t':3,'-':4
            } 
def one_d(idx_, d, max_len=2056*8):
    idx = idx_[None]
    K = torch.arange(d//2).to(idx.device)
    sin_e = torch.sin(idx[..., None] * math.pi / (max_len**(2*K[None]/d))).to(idx.device)
    cos_e = torch.cos(idx[..., None] * math.pi / (max_len**(2*K[None]/d))).to(idx.device)
    return torch.cat([sin_e, cos_e], axis=-1)[0]

def emb_alphas(alphas,seq_idx):
    data = [
        ("",' '.join(alphas)),
    ] 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract embeddings (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device),repr_layers=[12])
    token_embeddings = results["representations"][12][0,1:-1]
    return token_embeddings
def rnalm_alphas(alphas,seq_idx):
    seq  = ''.join(alphas)
    seqnpy=np.zeros(len(seq),dtype=int) + lmaadic['-']
    seq1=np.array(list(seq))  
    keys = list(lmaadic.keys())
    for akey in keys:
        seqnpy[seq1==akey] = lmaadic[akey]
    seqnpy = np.eye(lmaadic['-']+1)[seqnpy]
    fea  = {'aa':seqnpy,'idx':seq_idx,'mask':np.zeros(len(seq))}
    with torch.no_grad():
        lms,lmz = RNAlm.embedding({ 
            'aa':torch.FloatTensor(fea['aa']).to(device ),
            'idx':seq_idx,
            'mask':torch.FloatTensor(fea['mask']).to(device ),
            })
    return lms,lmz
def batch_emb_alphas(alphas,seq_idx):
    L = len(alphas)
    data = [(str(i),' '.join(alphas)) for i in range(L)] 

    batch_labels, batch_strs, batch_tokenss = batch_converter(data)
    for i in range(L):
        batch_tokenss[i,i+1] = 24

    # Extract embeddings (on CPU)
    token_embeddings=[]
    batch_tokenss = batch_tokenss.to(device)
    with torch.no_grad():
        for batch_tokens in batch_tokenss:
            results = model(batch_tokens,repr_layers=[12])
            token_embeddings.append(results["representations"][12][0,1:-1])
    return torch.stack(token_embeddings)

class PreMSA(nn.Module):
    def __init__(self,seq_dim,msa_dim,m_dim,z_dim):
        super(PreMSA,self).__init__()
        self.msalinear=basic.Linear(msa_dim,m_dim)
        self.qlinear  =basic.Linear(seq_dim,z_dim)
        self.klinear  =basic.Linear(seq_dim,z_dim)
        self.slinear  =basic.Linear(seq_dim,m_dim)
        self.pos = self.compute_pos().float()
        self.pos1d=self.compute_apos()
        self.poslinear=basic.Linear(64,z_dim)
        self.poslinear2=basic.Linear(64,m_dim)

        self.fm_layer = basic.Linear(640,m_dim)
        self.lm_layer_s = basic.Linear(512,m_dim)
        self.lm_layer_z = basic.Linear(128,z_dim)
    def tocuda(self,device):
        self.to(device)
        self.pos.to(device)
    def compute_apos(self,maxL=2000):
        d = torch.arange(maxL)
        m = 14
        d =(((d[:,None] & (1 << np.arange(m)))) > 0).float()
        return d

    def compute_pos(self,maxL=2000):
        a = torch.arange(maxL)
        b = (a[None,:]-a[:,None]).clamp(-32,32)
        return F.one_hot(b+32,65)


    def forward(self,seq,msa,idx,alphas):
        if self.pos.device != msa.device:
            self.pos = self.pos.to(msa.device)
        if self.pos1d.device != msa.device:
            self.pos1d = self.pos1d.to(msa.device)
        # msa N L D, seq L D
        N,L,D=msa.shape
        s = self.slinear(seq)
        m = self.msalinear(msa)
        p = self.poslinear2( one_d(idx, 64)   )

        fm_idx =   F.pad(idx, (1,1), "constant", 0) 
        fm_idx[-1] = idx[-1] + 1

        #s_fm = emb_alphas(alphas,fm_idx+1)
        s_lm,z_lm = rnalm_alphas(alphas,idx)
        m = m + s[None,:,:] + p[None,:,:] #+ self.fm_layer(s_fm)[None,:,:]
        m = m + self.lm_layer_s(s_lm)[None]
        

        sq=self.qlinear(seq)
        sk=self.klinear(seq)
        z=sq[None,:,:]+sk[:,None,:]

        seq_idx = idx[None]
        relative_pos = seq_idx[:, :, None] - seq_idx[:, None, :]
        relative_pos = relative_pos.reshape([1, L**2])
        relative_pos =one_d(relative_pos,64)

        z = z + self.poslinear( relative_pos.reshape([1, L, L, -1])[0] )
        z = z + self.lm_layer_z(z_lm)
        return m,z

def fourier_encode_dist(x, num_encodings = 20, include_self = True):
    # from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x



class RecyclingEmbedder(nn.Module):
    def __init__(self,m_dim,z_dim,dis_encoding_dim):
        super(RecyclingEmbedder,self).__init__()  
        self.linear = basic.Linear(dis_encoding_dim*2+1,z_dim)
        self.dis_encoding_dim=dis_encoding_dim
        self.normz = nn.LayerNorm(z_dim)
        self.normm = nn.LayerNorm(m_dim)
        self.dist_linear = nn.Sequential(
                        nn.Linear(36+2, z_dim),
                        nn.ReLU(),
                        nn.Linear(z_dim, z_dim),
                    )
        self.hb_linear = nn.Sequential(
                        nn.Linear(6, z_dim),
                        nn.ReLU(),
                        nn.Linear(z_dim, z_dim),
                    )
    def forward(self,m,z,x,previous_dist,previous_hb,first):
        cb = x[:,-1]
        dismap=(cb[:,None,:]-cb[None,:,:]).norm(dim=-1)
        dis_z = fourier_encode_dist(dismap,self.dis_encoding_dim)
        if first:
            return 0,self.linear(dis_z)   
        else:
            z = self.normz(z) + self.linear(dis_z)   + self.dist_linear(previous_dist) +  self.hb_linear(previous_hb)
            m = self.normm(m)
            return m,z 
        


class zBlock(nn.Module):
    def __init__(self,z_dim):
        super(zBlock,self).__init__()
        self.pair_triout=EvoPair.TriOut(z_dim)
        self.pair_triin =EvoPair.TriIn(z_dim)
        self.pair_tristart=EvoPair.TriAttStart(z_dim)
        self.pair_triend  =EvoPair.TriAttEnd(z_dim)
        self.pair_trans = EvoPair.PairTrans(z_dim)
    def forward(self,z):
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_tristart(z)
        z = z + self.pair_triend(z)
        z = z + self.pair_trans(z)
        return z
class ssAttention(nn.Module):
    def __init__(self, z_dim,N_head=8,c=8) -> None:
        super(ssAttention,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(z_dim)
        self.qlinear = basic.LinearNoBias(z_dim,N_head*c)
        self.klinear = basic.LinearNoBias(z_dim,N_head*c)
        self.vlinear = basic.LinearNoBias(z_dim,N_head*c)

        self.glinear = basic.Linear(z_dim,N_head*c)
        self.olinear = basic.Linear(N_head*c,z_dim)
    def forward(self,z):
        N,L,_,D = z.shape
        z = self.norm1(z)
        q = self.qlinear(z).reshape(N,L,L,self.N_head,self.c) 
        k = self.klinear(z).reshape(N,L,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(z).reshape(N,L,L,self.N_head,self.c)
        g = torch.sigmoid(self.glinear(z)).reshape(N,L,L,self.N_head,self.c)
        att = torch.einsum('iabhc,jabhc->ijabh',q,k)* (self.sq_c)
        att=F.softmax(att,dim=1)
        o = torch.einsum('ijabh,jabhc->iabhc',att,v) * g
        z = self.olinear(o.reshape(N,L,L,-1))   
        return z
class ssModule(nn.Module):
    def __init__(self,z_dim,N_head=8,c=8):
        super(ssModule,self).__init__()
        self.z_dim = z_dim
        self.emblinear = nn.Linear(1,z_dim)
        self.block1 = zBlock(z_dim)
        self.block2 = zBlock(z_dim)
        self.ssatt1  = ssAttention(z_dim)
        self.trans1  = EvoPair.PairTrans(z_dim,1)
        self.ssatt2  = ssAttention(z_dim)
        self.trans2  = EvoPair.PairTrans(z_dim,1)
        self.ssatt3  = ssAttention(z_dim)
        self.trans3  = EvoPair.PairTrans(z_dim,1)
    def batchz(self,z,bk):
        N = z.shape[0]
        slist = [bk(z[n]) for n in range(N)]
        return torch.stack(slist)
        


    def forward(self,ss):
        # ss N x l x l 
        z = self.emblinear(ss[...,None]) # N x l x l  x z
        z = self.batchz(z,self.block1)
        #z = self.batchz(z,self.block2) # N x l x l  x z
        z = z + self.ssatt1(z)
        z = z + self.trans1(z)
        z = z + self.ssatt2(z)
        z = z + self.trans2(z)
        z = z + self.ssatt3(z)
        z = z + self.trans3(z)
        return z



class MSA2xyzIteration(nn.Module):
    def __init__(self,seq_dim,msa_dim,N_ensemble,m_dim=64,s_dim=128,z_dim=64,docheck=True):
        super(MSA2xyzIteration,self).__init__()
        self.msa_dim=msa_dim
        self.m_dim=m_dim
        self.z_dim=z_dim
        self.seq_dim=seq_dim
        self.N_ensemble=N_ensemble
        self.dis_dim=36 + 2 
        self.pre_z=ssModule(z_dim)
        self.premsa=PreMSA(seq_dim,msa_dim,m_dim,z_dim)
        self.re_emb=RecyclingEmbedder(m_dim,z_dim,dis_encoding_dim=64)
        #self.ex_emb = RecyclingPoolEmbedder(m_dim = m_dim,s_dim = s_dim,z_dim = z_dim,dis_encoding_dim=32, dis_dim = self.dis_dim)
        self.evmodel=Evoformer.Evoformer(m_dim,z_dim,True)    
        self.slinear=basic.Linear(z_dim,s_dim)

    def pred(self,msa_,idx,ss_,m1_pre,z_pre,pre_x,cycle_index,alphas,previous_dis,previous_hb):
        m1_all,z_all,s_all=0,0,0
        N,L,_=msa_.shape
        for i in range(self.N_ensemble):
            msa_mask = torch.zeros(N,L).to(msa_.device)
            msa_true = msa_ + 0
            seq = msa_true[0]*1.0 # 22-dim
            msa = torch.cat([msa_true*(1-msa_mask[:,:,None]),msa_mask[:,:,None]],dim=-1)
            m,z=self.premsa(seq,msa,idx,alphas)
            if ss_ is None:
                ss = 0
            else:
                ss = torch.mean( self.pre_z(ss_),dim=0)
            #ss = self.pre_z(ss_)
            z  = z+ss
            m1_,z_=self.re_emb(m1_pre,z_pre,pre_x,previous_dis,previous_hb,cycle_index==0) #already added residually
            #ex_s,ex_z =self.ex_emb(previous_s,previous_z,previous_dis,previous_x,previous_hb)
            z = z+z_
            m=torch.cat([(m[0]+m1_)[None,...],m[1:]],dim=0)
            m,z=self.evmodel(m,z)
            s = self.slinear(m[0])
            m1_all =m1_all + m[0]
            z_all  =z_all  + z
            s_all  =s_all + s
        return m1_all/self.N_ensemble,z_all/self.N_ensemble,s_all/self.N_ensemble


class MSA2XYZ(nn.Module):
    def __init__(self,seq_dim,msa_dim,N_ensemble,N_cycle,m_dim=64,s_dim=128,z_dim=64,docheck=True):
        super(MSA2XYZ,self).__init__()
        self.msa_dim=msa_dim
        self.m_dim=m_dim
        self.z_dim=z_dim
        self.dis_dim=36 + 2
        self.N_cycle=N_cycle
        self.msaxyzone = MSA2xyzIteration(seq_dim,msa_dim,N_ensemble,m_dim=m_dim,s_dim=s_dim,z_dim=z_dim)
        self.msa_predor=basic.Linear(m_dim,msa_dim-1)
        self.pdis_predor=basic.Linear(z_dim,self.dis_dim)
        self.cdis_predor=basic.Linear(z_dim,self.dis_dim)
        self.ndis_predor=basic.Linear(z_dim,self.dis_dim)
        self.hb_predor=basic.Linear(z_dim,6)
        self.slinear=basic.Linear(m_dim,s_dim)
        
        self.structurenet=Structure.StructureModule(s_dim,z_dim,4,s_dim) #s_dim,z_dim,N_layer,c)

    def pred(self,msa_,idx,ss,base_x,alphas):
        plddts =[]
        predxs={}
        L=msa_.shape[1]
        m1_pre,z_pre=0,0
        x_pre=torch.zeros(L,3,3).to(msa_.device)
        previous_dis = torch.zeros([L,L,38]).to(msa_.device) 
        previous_dis[...,0] = 1
        previous_hb = torch.zeros([L,L,6]).to(msa_.device)
        previous_hb[...,0] = 1
        ret = {}
        for i in range(self.N_cycle):
            m1,z,s=self.msaxyzone.pred(msa_,idx,ss,m1_pre,z_pre,x_pre,i,alphas,previous_dis,previous_hb)
            x,_,_,plddt = self.structurenet.pred(s,z,base_x)
            pred_disn = F.softmax(self.ndis_predor(z),dim=-1)  
            pred_hb   = F.sigmoid(self.hb_predor(z)) 
            #plddts.append(plddt)
            m1_pre=m1.detach()
            z_pre = z.detach()
            x_pre = x.detach()
            predxs[i]=x_pre.cpu().detach()
            previous_dis=pred_disn.detach()
            previous_hb=pred_hb.detach()
            ret['coor'] = x_pre.detach().cpu().numpy()
            ret['dist_p'] = F.softmax(self.pdis_predor(z),dim=-1).detach().cpu().numpy().astype(np.float16)  
            ret['dist_c'] = F.softmax(self.cdis_predor(z),dim=-1).detach().cpu().numpy().astype(np.float16)
            ret['dist_n'] = F.softmax(self.ndis_predor(z),dim=-1).detach().cpu().numpy().astype(np.float16)
            ret['plddt'] = plddt.detach().cpu().numpy()
        ########################last cycle###########

        return ret













