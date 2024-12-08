import os,sys
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import torch
exp_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"



dlexps = ['cfg_95','cfg_96','cfg_97','cfg_99']


fastafile =  os.path.realpath(sys.argv[1])
outdir = os.path.realpath(sys.argv[2])



pclu    = False
if len(sys.argv) == 4 and sys.argv[3] == '1': 
    print('will do cluster')
    pclu = True








if not os.path.isdir(outdir):
    os.makedirs(outdir)
ret_dir = os.path.join(outdir,'rets_dir')
if not os.path.isdir(ret_dir):
    os.makedirs(ret_dir)


folddir = os.path.join(outdir,'folds')
if not os.path.isdir(folddir):
    os.makedirs(folddir)

refdir = os.path.join(outdir,'relax')
if not os.path.isdir(refdir):
    os.makedirs(refdir)


dlmains = [os.path.join(exp_dir,one_exp,'test_modeldir.py') for one_exp in dlexps]
dirs = [os.path.join(exp_dir,'model_hub',one_exp) for one_exp in dlexps]
if not os.path.isfile(ret_dir+'/done'):
    print(ret_dir+'/done', 'is not here. Will generate e2e and geo files.')
    for dlmain,one_exp,mdir in zip(dlmains,dlexps,dirs):
        cmd = f'python {dlmain} {device} {fastafile} {ret_dir}/{one_exp}_ {mdir}'
        print(cmd)
        # expdir=os.path.dirname(os.path.abspath(__file__))
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output,error = p.communicate()
        #print(output,error)
    wfile = open(ret_dir+'/done','w')
    wfile.write('1')
    wfile.close()
else:
    print(ret_dir+'/done', 'is here, using existing e2e and geo files.')


def get_model_pdb(tdir,opt):
    files = os.listdir(tdir)
    files = [afile for afile in files if afile.startswith(opt)][0]
    return files

cso_dir = folddir
clufile = os.path.join(folddir,'clu.txt')
config_sel = os.path.join(exp_dir,'cfg_for_selection.json')
foldconfig = os.path.join(exp_dir,'cfg_for_folding.json')
selpython = os.path.join(exp_dir,'PotentialFold','Selection.py')
optpython = os.path.join(exp_dir,'PotentialFold','Optimization.py')
clupy = os.path.join(exp_dir,'PotentialFold','Clust.py')
arena = os.path.join(exp_dir,'Arena','Arena')

optsaveprefix=os.path.join(cso_dir,f'opt_0')
save_prefix = os.path.join(cso_dir,f'sel_0')
rets = os.listdir(ret_dir)
rets = [afile for afile in rets if afile.endswith('.ret')]
rets = [os.path.join(ret_dir,aret) for aret in rets ]
ret_str = ' '.join(rets)
cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
output,error = p.communicate()
#print(output,error)
cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
output,error = p.communicate()
#print(output,error)
cgpdb = os.path.join(folddir,get_model_pdb(folddir,'opt_0'))
savepdb = os.path.join(refdir,'model_1.pdb')
cmd = f'{arena} {cgpdb} {savepdb} 7'
p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
output,error = p.communicate()
#print(output,error)

if pclu:
    cmd = f'python {clupy} {ret_dir} {clufile}'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output,error = p.communicate()


    lines = open(clufile).readlines()
    lines = [aline.strip() for aline in lines]
    lines = [aline for aline in lines if aline]

    for i in range(1,len(lines)):
        rets = lines[i].split()
        rets = [os.path.join(ret_dir,aret.replace('.pdb','.ret')) for aret in rets ]
        ret_str = ' '.join(rets)
        optsaveprefix =  os.path.join(cso_dir,f'opt_{str(i+1)}')
        save_prefix = os.path.join(cso_dir,f'sel_{str(i+1)}')
        cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
        print(cmd)
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output,error = p.communicate()
        #print(output,error)
        cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output,error = p.communicate()
        #print(output,error)
        cgpdb = os.path.join(folddir,get_model_pdb(folddir,f'opt_{str(i+1)}'))
        savepdb = os.path.join(refdir,f'model_{str(i+1)}.pdb')
        cmd = f'{arena} {cgpdb} {savepdb} 7'
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output,error = p.communicate()
