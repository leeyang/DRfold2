import torch
import numpy as np 
from subprocess import Popen, PIPE, STDOUT
import os,sys



def outpdb(coor,seq_idx,seq,savefile,start=0,end=10000,energystr=''):
    #rama=torch.cat([rama.view(self.L,2),self.betas],dim=-1)
    L = coor.shape[0]
    
    Atom_name=[' P  '," C4'",' N1 ']
    Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
    other_last_name = ['O',"C","C","O","C"]


    last_name=['P','C','N']
    wstr=[f'REMARK {str(energystr)}']
    templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
    count=1
    for i in range(L):
        if seq[i] in ['a','g','A','G']:
            Atom_name = [' P  '," C4'",' N9 ']
            #atoms = ['P','C4']

        elif seq[i] in ['c','u','C','U']:
            Atom_name = [' P  '," C4'",' N1 ']
        for j in range(coor.shape[1]):
            outs=('ATOM  ',count,Atom_name[j],seq[i],'A',seq_idx[i],coor[i][j][0],coor[i][j][1],coor[i][j][2],0,0,last_name[j],'')
            #outs=('ATOM  ',count,Atom_name[j],'ALA','A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],1.0,90,last_name[j],'')
            #print(outs)
            if i>=start-1 and i < end:
                wstr.append(templet % outs)

        # for j in range(other_np.shape[1]):
        #     outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
        #     #outs=('ATOM  ',count,Atom_name[j],'ALA','A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],1.0,90,last_name[j],'')
        #     #print(outs)
        #     if i>=start-1 and i < end:
        #         wstr.append(templet % outs)
            count+=1
    wstr.append('TER')  
    wstr='\n'.join(wstr)
    wfile=open(savefile,'w')
    wfile.write(wstr)
    wfile.close()



