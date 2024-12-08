import torch
import os,math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import sample
from numpy import float32
import random
from scipy.spatial.distance import cdist
from subprocess import Popen, PIPE, STDOUT
expdir=os.path.dirname(os.path.abspath(__file__))

code_standard = {
'A':'A','G':'G','C':'C','U':'U','a':'A','g':'G','c':'C','u':'U','T':'U','t':'U'
} 
expdir=os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(expdir)



        





def parse_seq(inseq):
    seqnpy=np.zeros(len(inseq))
    seq1=np.array(list(inseq))
    seqnpy[seq1=='A']=1
    seqnpy[seq1=='G']=2
    seqnpy[seq1=='C']=3
    seqnpy[seq1=='U']=4
    seqnpy[seq1=='T']=4
    return seqnpy




def Get_base(seq,basenpy_standard):
    basenpy = np.zeros([len(seq),3,3])
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
    return basenpy






