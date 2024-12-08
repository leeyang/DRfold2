import os,sys
import numpy as np 
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
cif_parser = MMCIFParser(QUIET=True)
pdb_parser = PDBParser(QUIET=True)
def compputeU(x_,y_):
    with torch.no_grad():
        # x_.shape:[N,3]
        xm,ym=x_.mean(dim=0,keepdim=True),y_.mean(dim=0,keepdim=True)
        x=x_-xm
        y=y_-ym
        r=torch.matmul(x.T,y).T
        
        t=torch.empty([4,4],device=x_.device)
        t[0,0]=(r[0,0]+r[1,1]+r[2,2])
        t[0,1]=r[1,2]-r[2,1]
        t[0,2]=r[2,0]-r[0,2]
        t[0,3]=r[0,1]-r[1,0]
        t[1,0]=t[0,1]
        t[1,1]=(r[0,0]-r[1,1]-r[2,2])
        t[1,2]=r[0,1]+r[1,0]
        t[1,3]=r[0,2]+r[2,0]
        t[2,0]=t[0,2]
        t[2,1]=t[1,2]
        t[2,2]=(r[1,1]-r[0,0]-r[2,2])
        t[2,3]=r[1,2]+r[2,1]
        t[3,0]=t[0,3]
        t[3,1]=t[1,3]
        t[3,2]=t[2,3]
        t[3,3]=(r[2,2]-r[0,0]-r[1,1])
        #t=(t+t.T)/2
        #print(t)
        ws, q = torch.linalg.eigh(t)
        #print(ws)
        w,q=ws[-1],(q.T)[-1]
        u=torch.empty([3,3],device=x_.device)
        u[0,0]=q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3]
        u[0,1]=2*(q[1]*q[2]-q[0]*q[3])
        u[0,2]=2*(q[1]*q[3]+q[0]*q[2])
        u[1,0]=2*(q[1]*q[2]+q[0]*q[3])
        u[1,1]=q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3]
        u[1,2]=2*(q[2]*q[3]-q[0]*q[1])
        u[2,0]=2*(q[1]*q[3]-q[0]*q[2])
        u[2,1]=2*(q[2]*q[3]+q[0]*q[1])
        u[2,2]=q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]
        return u,xm,ym

def get_des(x_,y_,u,xm,ym):
    with torch.no_grad():
        y=y_-ym
        y=torch.matmul(y,u.T)+xm
        return y

def compute_RMSD(coor1_,coor2_):
    u,xm,ym=compputeU(coor1_,coor2_)
    x_des=get_des(coor1_,coor2_,u,xm,ym)
    L=coor1_.shape[0]
    residue=((coor1_-x_des)**2).sum(-1)
    rmsd=torch.sqrt((residue).mean())
    # tmscore=torch.sqrt(residue)
    # d0=1.24*(L**(1.0/3.0))-1.8
    # tmscore=1.0/(1+(tmscore/d0)**2  )
    return rmsd#,tmscore.mean()
def read_C4(pdb_file):
    if pdb_file.endswith('.pdb'):
        structure = pdb_parser.get_structure('',pdb_file)
    elif pdb_file.endswith('.cif'):
        structure = cif_parser.get_structure('',pdb_file)
    else:
        print(pdb_file,'file type unrecongnized!')
        assert False
    coors=[]
    model = structure[0]
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom_name = atom.name
                #print(atom_name)
                if atom_name in ["C4'"]:
                    coors.append(atom.coord)
    return np.array(coors)


def cal_RMSD(coor1,coor2):
    coor1_ = torch.from_numpy(coor1)
    coor2_ = torch.from_numpy(coor2)
    rmsd = compute_RMSD(coor1_,coor2_).item()
    return rmsd



def get_coors_from_dir(adir):
    pdbs = os.listdir(adir)
    pdbs = [apdb for apdb in pdbs if apdb.endswith('.pdb')]
    pdbs.sort()
    #print(pdbs)
    coors = [read_C4(os.path.join(adir,apdb)) for apdb in pdbs]
    nump = len(coors)
    rmsd_map = np.zeros([nump,nump])
    rmsds=[]
    for i in range(nump):
        for j in range(nump):
            rmsd_map[i,j] = cal_RMSD(coors[i],coors[j])
            if rmsd_map[i,j] >0.5:
                rmsds.append(rmsd_map[i,j] )
    print(np.min(rmsds), np.mean(rmsds), np.max(rmsds))
    return rmsd_map,pdbs ,np.min(rmsds), np.mean(rmsds), np.max(rmsds)

def op_dcutN(crmsd_map,dcut):
    #L,_ = rmsd_map.shape[0]
    count_bin = (crmsd_map < dcut)
    count = count_bin.sum(axis=-1)
    maxidx = count.argmax()
    return maxidx, count_bin[maxidx], count[maxidx]


def check_NDmax(N,L,dcut,max_percent,min_dcut):
    return (float(N/L)  > max_percent) and (dcut>min_dcut)
def check_NDmiin(N,L,dcut,min_percent,max_dcut):
    return (float(N/L)  < min_percent) and (dcut<max_dcut)
def cluster(rmsd_map, dcut = 30,max_percent = 0.7, min_percent = 0.15):
    max_dcut = 40
    min_dcut = 25
    currentdcut = dcut
    L = rmsd_map.shape[0]
    while True:
        maxidx,maxn = op_dcutN(rmsd_map,currentdcut)
        if check_NDmax(maxn,L,currentdcut,max_percent,min_dcut):
            currentdcut = currentdcut - 1
        else:
            if check_NDmiin(maxn,L,currentdcut,min_percent,max_dcut):
                currentdcut = currentdcut + 1
            else:
                pass

        
        

class scricker:
    def __init__(self,rmsd_map,pdbs,rmin,rmean,rmax) -> None:
        self.dcut = (rmean - rmin)*0.4 + rmin
        self.max_dcut = (rmax - rmean )*0.15 + rmean
        self.min_dcut = (rmean - rmin)*0.15 + rmin
        self.max_percent = 0.5
        self.min_percent = 0.15

        self.rmsd_map = rmsd_map
        self.L = rmsd_map.shape[0]
        self.pdbs = pdbs

    def check_NDmax(self,N,curretndcut):
        return (float(N/self.L)  > self.max_percent) and (curretndcut>self.min_dcut)
    def check_NDmiin(self,N,curretndcut):
        return (float(N/self.L)  < self.min_percent) and (curretndcut<self.max_dcut)

    def find_max_clu(self,current_rmsd_map, node_names,currentdcut):
        while True:
            maxidx,max_ids,maxn = op_dcutN(current_rmsd_map,currentdcut)
            
            if self.check_NDmax(maxn,currentdcut):
                currentdcut = currentdcut - 0.2
                #print(maxn,currentdcut)
            else:
                if self.check_NDmiin(maxn,currentdcut):
                    currentdcut = currentdcut + 0.2
                    #print(maxn,currentdcut)
                else:
                    #print(max_ids,node_names)
                    return    node_names[max_ids]  ,currentdcut,node_names[maxidx]


    def main_loop(self):
        clusters = []
        currentdcut = self.dcut
        current_rmsd_map = self.rmsd_map
        current_names = np.arange(self.L)
        clusters_files=[]
        clu_lists = []
        for i in range(5):
            #print('current_rmsd_map.shape',current_rmsd_map.shape)
            idxes,currentdcut,centreid = self.find_max_clu(current_rmsd_map, current_names,currentdcut)
            clusters.append(idxes)
            current_names = [onename for onename in current_names if onename not in idxes]
            clusters_files.append(self.pdbs[centreid])
            print(len(current_names))
            current_rmsd_map = self.rmsd_map[current_names][:,current_names]
            current_names = np.array(current_names)
            print(idxes,currentdcut,[self.pdbs[aid] for aid in idxes],centreid,self.pdbs[centreid])
            clu_lists.append( [self.pdbs[aid] for aid in idxes] )
            if len(current_names) <1:
                break
        return clusters_files,clu_lists


#coors=read_C4('R1286/rets/cfg_97_epoch_522000.pdb')
rmsd,pdbs,rmin,rmean,rmax = get_coors_from_dir(sys.argv[1])
clu = scricker(rmsd,pdbs,rmin,rmean,rmax)
clusters_files,clu_lists=clu.main_loop()
wfile = open(sys.argv[2],'w')
# wfile.write('\n'.join(clusters_files))
# wfile.close()
for alist in clu_lists:
    wfile.write(' '.join(alist)+'\n')
wfile.close()
