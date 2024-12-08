import random
random.seed(0)
import numpy as np
np.random.seed(0)
import os,sys,re,random
from numpy import select
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
expdir=os.path.dirname(os.path.abspath(__file__))

# from pathlib import Path
# path = Path(expdir)
# parepath = path.parent.absolute()


import torch.optim as opt
from torch.nn import functional as F
import data,util
import EvoMSA2XYZ,basic
import math
import pickle
Batch_size=3
Num_cycle=3
TEST_STEP=1000
VISION_STEP=50
device = sys.argv[1]


expdir=os.path.dirname(os.path.abspath(__file__))
expround=expdir.split('_')[-1]
model_path=os.path.join(expdir,'others','models')

testdir=os.path.join(expdir,'others','preds')




basenpy_standard= np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'base.npy'  )  )
def data_collect(pdb_seq):
    aa_type = data.parse_seq(pdb_seq)
    base = data.Get_base(pdb_seq,basenpy_standard)
    seq_idx = np.arange(len(pdb_seq)) + 1

    msa=aa_type[None,:]
    msa=torch.from_numpy(msa).to(device)
    msa=torch.cat([msa,msa],0)
    msa=F.one_hot(msa.long(),6).float()

    base_x = torch.from_numpy(base).float().to(device)
    seq_idx = torch.from_numpy(seq_idx).long().to(device)
    return msa,base_x,seq_idx
    predxs,plddts = model.pred(msa,seq_idx,ss,base_x,sample_1['alpha_0'])



def classifier(infasta,out_prefix,model_dir):
    with torch.no_grad():
        lines = open(infasta).readlines()[1:]
        seqs = [aline.strip() for aline in lines]
        seq = ''.join(seqs)
        msa,base_x,seq_idx = data_collect(seq)
        # seq_idx = np.genfromtxt(idxfile).astype(int)
        # seq_idx = torch.from_numpy(seq_idx).long().to(device)
        
        msa_dim=6+1
        m_dim,s_dim,z_dim = 64,64,64
        N_ensemble,N_cycle=3,8
        model=EvoMSA2XYZ.MSA2XYZ(msa_dim-1,msa_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim)
        model.to(device)
        model.eval()
        models = os.listdir(  model_dir   )
        models = [amodel for amodel in models if 'model' in amodel and 'opt' not in amodel]

        models.sort()
        # models = models[5:]

        for amodel in models:
            #saved_model=os.path.join(expdir,'others','models',amodel)
            saved_model=os.path.join(model_dir,amodel)
            model.load_state_dict(torch.load(saved_model,map_location='cpu'),strict=True)
            ret = model.pred(msa,seq_idx,None,base_x,np.array(list(seq)))

            util.outpdb(ret['coor'],seq_idx,seq,out_prefix+f'{amodel}.pdb')
            #ret = {'plddt':ret['plddt']}
            pickle.dump(ret,open(out_prefix+f'{amodel}.ret','wb'))
            # for akey in ret:
            #     print(akey,ret[akey].shape)


 
 
    

if __name__ == '__main__':
    infasta,out_prefix,model_dir = sys.argv[2],sys.argv[3],sys.argv[4]
    classifier(infasta,out_prefix,model_dir)