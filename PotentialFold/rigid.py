import torch


def C4P_C3P(bp):
    if bp in ['A','a']:
        return torch.FloatTensor([0.4219, 0.7000, 1.2829])
    if bp in ['G','g']:
        return torch.FloatTensor([0.4240, 0.6989, 1.2829])
    if bp in ['C','c']:
        return torch.FloatTensor([0.4567, 0.6948, 1.2731])
    if bp in ['U','u']:
        return torch.FloatTensor([0.4556, 0.6952, 1.2736])



def C3P_O3P(bp):
    if bp in ['A','a']:
        return torch.FloatTensor([0.3506,  1.3587, -0.1989])
    if bp in ['G','g']:
        return torch.FloatTensor([0.3519,  1.3587, -0.2005])
    if bp in ['C','c']:
        return torch.FloatTensor([0.3747,  1.3505, -0.2116])
    if bp in ['U','u']:
        return torch.FloatTensor([0.3747,  1.3497, -0.2093])


def base_table():
    base_dict={}
    base_dict['atoms']                      =['N1','C2','O2','N2','N3','N4','C4','O4','C5','C6','O6','N6','N7','C8','N9']
    base_dict['a_mask']=torch.FloatTensor(   [ 1  ,   1,   0,   0,   1,    0,   1,   0,  1,    1,   0,   1,   1,  1,   1])
    base_dict['g_mask']=torch.FloatTensor(   [ 1  ,   1,   0,   1,   1,    0,   1,   0,  1,    1,   1,   0,   1,  1,   1])
    base_dict['c_mask']=torch.FloatTensor(   [ 1  ,   1,   1,   0,   1,    1,   1,   0,  1,    1,   0,   0,   0,  0,   0])
    base_dict['u_mask']=torch.FloatTensor(   [ 1  ,   1,   1,   0,   1,    0,   1,   1,  1,    1,   0,   0,   0,  0,   0])
    return base_dict


def side_mask(seq):
    base_dict = base_table()
    masks=[]
    for bp in seq:
        masks.append(  torch.cat( [torch.FloatTensor([1]*7),base_dict[bp.lower()+'_mask']] ,dim=0)  )

    return torch.stack(masks,dim=0)
        
