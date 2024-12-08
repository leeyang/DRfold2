import numpy as np 
from scipy.interpolate import CubicSpline,UnivariateSpline
import os
from torch.autograd import Function
import torch
import math,json


def batched_index_select(input, dim, index):
    # https://discuss.pytorch.org/t/batched-index-select/9115/8
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index).squeeze()

class cubic_batch_dis_class(Function):
    @staticmethod
    def forward(ctx,input1,coe,x,min_dis,max_dis,bin_num):
        # inoput: B coe: B 3 i
        # min_ref=config['min_dis']+((config['max_dis']-config['min_dis'])/config['bin_num'])*1.5
        # bin_size = (config['max_dis']-config['min_dis'])/config['bin_num']
        min_ref = min_dis+((max_dis-min_dis)/bin_num)*1.5
        bin_size = (max_dis-min_dis)/bin_num
        inputi= input1.detach()
        selction1=inputi <=min_ref
        selction2=inputi > min_ref
        ctx.selction1=selction1
        ctx.selction2=selction2   
        ctx.inputi=inputi
        ctx.coe=coe
        out=inputi*1.0
        #out[selction1] = coe[selction1,0,0]*(inputi[selction1]+1e-4)**3 + coe[selction1,1,0]*(inputi[selction1]+1e-4)**2 + coe[selction1,2,0]*(inputi[selction1]+1e-4) + coe[selction1,3,0]
        #indexes =(( (inputi[selction2]-min_ref) // 0.5) +1).long()
        indexes = (torch.div((inputi[selction2]-min_ref), bin_size, rounding_mode='floor') +1 ).long()
        selectedaoe = batched_index_select(coe[selction2],2,indexes) # B 3 
        selectedx= batched_index_select(x[selction2],1,indexes) # B 
        input2=inputi[selction2]
        out[selction2] = selectedaoe[:,0]*(input2-selectedx)**3 + selectedaoe[:,1]*(input2-selectedx)**2 + selectedaoe[:,2]*(input2-selectedx) + selectedaoe[:,3]
        ctx.indexes=indexes
        ctx.selectedx=selectedx
        ctx.selectedaoe =selectedaoe
        return out
    @staticmethod
    def backward(ctx,grad_output):
        inputi=ctx.inputi
        grad = inputi+0.0
        grad[ctx.selction1] = 3*ctx.coe[ctx.selction1,0,0]*(inputi[ctx.selction1]+1e-4)**2 + 2*ctx.coe[ctx.selction1,1,0]*(ctx.inputi[ctx.selction1]+1e-4) + ctx.coe[ctx.selction1,2,0]
        grad[ctx.selction2] = 3*ctx.selectedaoe[:,0]*(inputi[ctx.selction2]- ctx.selectedx)**2 + 2*ctx.selectedaoe[:,1]*(ctx.inputi[ctx.selction2]-ctx.selectedx) + ctx.selectedaoe[:,2]
        return grad_output*grad,None,None,None,None,None,None
def cubic_distance(input1,coe,x,min_dis,max_dis,bin_num):
    return cubic_batch_dis_class.apply(input1,coe,x,min_dis,max_dis,bin_num)


class cubic_batch_torsion_class(Function):
    @staticmethod
    def forward(ctx,input1,coe,x,num_bin):
        # inoput: B coe: B 3 i
        x0=x[0][0]
        inputi= input1.detach()
        inputi[inputi < x0] +=math.pi*2
        ctx.inputi=inputi
        ctx.coe=coe
        out=inputi*1.0
        #indexes =(( (inputi-x0) // (2*math.pi/num_bin))).long()
        indexes =torch.div((inputi-x0), (2*math.pi/num_bin), rounding_mode='floor').long()
        selectedaoe = batched_index_select(coe,2,indexes) # B 3 
        selectedx= batched_index_select(x,1,indexes) # B 

        out = selectedaoe[:,0]*(inputi-selectedx)**3 + selectedaoe[:,1]*(inputi-selectedx)**2 + selectedaoe[:,2]*(inputi-selectedx) + selectedaoe[:,3]
        ctx.indexes=indexes
        ctx.selectedx=selectedx
        ctx.selectedaoe =selectedaoe
        #print(out.shape)
        #print(out)
        return out
    @staticmethod
    def backward(ctx,grad_output):
        inputi=ctx.inputi
        grad = inputi+0.0
        
        grad = 3*ctx.selectedaoe[:,0]*(inputi- ctx.selectedx)**2 + 2*ctx.selectedaoe[:,1]*(ctx.inputi-ctx.selectedx) + ctx.selectedaoe[:,2]

        #print('grad',grad.shape)
        #print(grad.sum())
        return grad_output*grad,None,None,None
def cubic_torsion(input1,coe,x,num_bin):
    return cubic_batch_torsion_class.apply(input1,coe,x,num_bin)


class cubic_batch_angle_class(Function):
    @staticmethod
    def forward(ctx,input1,coe,x,num_bin=12):
        # inoput: B coe: B 3 i
        x0=x[0][0]
        inputi= input1.detach()
        #print(x0)
        ctx.inputi=inputi
        ctx.coe=coe
        out=inputi*1.0
        indexes =(( (inputi-x0) // (math.pi/num_bin))).long()
        selectedaoe = batched_index_select(coe,2,indexes) # B 3 
        selectedx= batched_index_select(x,1,indexes) # B 

        out = selectedaoe[:,0]*(inputi-selectedx)**3 + selectedaoe[:,1]*(inputi-selectedx)**2 + selectedaoe[:,2]*(inputi-selectedx) + selectedaoe[:,3]
        ctx.indexes=indexes
        ctx.selectedx=selectedx
        ctx.selectedaoe =selectedaoe
        #print(out.shape)
        #print(out)
        return out
    @staticmethod
    def backward(ctx,grad_output):
        inputi=ctx.inputi
        grad = inputi+0.0
        
        grad = 3*ctx.selectedaoe[:,0]*(inputi- ctx.selectedx)**2 + 2*ctx.selectedaoe[:,1]*(ctx.inputi-ctx.selectedx) + ctx.selectedaoe[:,2]

        #print('grad',grad.shape)
        #print(grad.sum())
        return grad_output*grad,None,None,None
def cubic_angle(input1,coe,x,num_bin):
    return cubic_batch_angle_class.apply(input1,coe,x,num_bin)



def LJpotential(dis,th):
    #238325838
    r = ( (th+0.5) / (dis+0.5))**6
    return (r**2 - 2*r)
    #return torch.clamp((r**2 - 2*r),max=228325838) 
if __name__ == '__main__': 
    print(LJpotential(   torch.Tensor([0.])  ,2.5))