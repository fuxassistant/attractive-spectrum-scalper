import argparse
from .experiment import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--run',default='run_001');p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seeds',type=int,nargs='+',default=[61015,61016,61017]);a=p.parse_args();run(a.root,a.run,a.epochs,tuple(a.seeds))
