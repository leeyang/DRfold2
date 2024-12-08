import numpy as np 
from scipy.interpolate import CubicSpline,UnivariateSpline
import os
from torch.autograd import Function
import torch
import math



def fit_dis_cubic(dis_matrix,min_dis,max_dis,num_bin):
    dis_region=np.zeros(num_bin)
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      (dis_matrix[i,j,1:-1]+1e-8) / (dis_matrix[i,j,[-2]]+1e-8)              )
            x=dis_region
            x[0]=-0.0001
            y[0]= max(10,y[1]+4)
            cs= CubicSpline(x,y)
            decs=cs.derivative()
            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)
    return np.array(csnp),np.array(decsnp)

def dis_cubic(out,min_dis,max_dis,num_bin):
    print('fitting cubic distance')
    cs,decs=fit_dis_cubic(out,min_dis,max_dis,num_bin)
    return cs,decs



def cubic_matrix_torsion(dis_matrix,min_dis,max_dis,num_bin):
    dis_region=np.zeros(num_bin)
    bin_size=(max_dis-min_dis)/num_bin
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      dis_matrix[i,j,:-1]+1e-8             )
            x=dis_region
            x=np.append(x,x[-1]+bin_size)
            y=np.append(y,y[0])
            cs= CubicSpline(x,y,bc_type='periodic')
            decs=cs.derivative()
            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)
    return np.array(csnp),np.array(decsnp)
def torsion_cubic(out,min_dis,max_dis,num_bin):
    print('fitting cubic')
    cs,decs=cubic_matrix_torsion(out,min_dis,max_dis,num_bin)
    return cs,decs

def cubic_matrix_angle(dis_matrix,min_dis,max_dis,num_bin): # 0 - np.pi 12
    dis_region=np.zeros(num_bin)
    bin_size=(max_dis-min_dis)/num_bin
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      dis_matrix[i,j,:-1]+1e-8             )
            x=dis_region

            x=np.concatenate([[x[0]-bin_size*3,x[0]-bin_size*2,x[0]-bin_size], x,[x[-1]+bin_size,x[-1]+bin_size*2,x[-1]+bin_size*3]               ])
            y=np.concatenate([ [y[2],y[1],y[0]],y,[y[-1],y[-2],y[-3]]                                                                                                                    ])

            cs= CubicSpline(x,y)
            decs=cs.derivative()

            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)

    return np.array(csnp),np.array(decsnp)
def angle_cubic(out,min_dis,max_dis,num_bin):

    print('fitting angle cubic')
    cs,decs=cubic_matrix_angle(out,min_dis,max_dis,num_bin)

    return cs,decs