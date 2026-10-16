import argparse
from .experiment import run

def main():
    p=argparse.ArgumentParser(description='Causal LSTM trading gym: first week learn/select, second week frozen replay')
    p.add_argument('--root',default='.');p.add_argument('--run',default='run_001');p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args();run(a.root,a.run,a.epochs)

if __name__=='__main__':main()
