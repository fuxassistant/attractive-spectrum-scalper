"""Synthetic forward/backward/AdamW memory probes; no trading/model fitting.

Fresh process per case; capped PyTorch allocator protects desktop headroom.
Input dimensions include numeric values AND masks after float packing.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

CASES = {
    'current_m1':dict(kind='flat',batch=1,length=1380,dim=71,hidden=48,layers=1,amp=False),
    'two_days_seconds_fp32':dict(kind='flat',batch=1,length=165600,dim=256,hidden=128,layers=2,amp=False),
    'two_days_seconds_amp':dict(kind='flat',batch=1,length=165600,dim=256,hidden=128,layers=2,amp=True),
    'two_days_seconds_chunked_amp':dict(kind='flat',batch=1,length=165600,dim=256,hidden=128,layers=2,amp=True,chunk=8192),
    'two_days_seconds_chunked_b4_amp':dict(kind='flat',batch=4,length=165600,dim=256,hidden=128,layers=2,amp=True,chunk=8192),
    'two_days_all295_chunked_b2_amp':dict(kind='flat',batch=2,length=165600,dim=590,hidden=128,layers=2,amp=True,chunk=8192),
    'hybrid_b8_amp':dict(kind='hybrid',batch=8,length=900,dim=256,hidden=128,layers=2,slow_length=2760,slow_dim=128,slow_hidden=64,amp=True),
    'hybrid_b16_amp':dict(kind='hybrid',batch=16,length=900,dim=256,hidden=128,layers=2,slow_length=2760,slow_dim=128,slow_hidden=64,amp=True),
    'hybrid_b32_amp':dict(kind='hybrid',batch=32,length=900,dim=256,hidden=128,layers=2,slow_length=2760,slow_dim=128,slow_hidden=64,amp=True),
}


def worker(name,out):
    import gc
    import torch
    from torch import nn
    c=CASES[name]
    torch.set_num_threads(2);torch.manual_seed(618);torch.cuda.set_per_process_memory_fraction(.72)
    torch.backends.cudnn.benchmark=False
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.project=nn.Sequential(nn.Linear(c['dim'],128 if c['hidden']==128 else 64),nn.Tanh())
            self.fast=nn.LSTM(128 if c['hidden']==128 else 64,c['hidden'],c['layers'],batch_first=True)
            if c['kind']=='hybrid':
                self.slow_project=nn.Sequential(nn.Linear(c['slow_dim'],64),nn.Tanh())
                self.slow=nn.LSTM(64,c['slow_hidden'],batch_first=True)
            self.head=nn.Linear(c['hidden']+(c.get('slow_hidden',0)),11)
        def forward(self,x,s=None):
            if c.get('chunk'):
                state=None
                for i in range(0,x.shape[1],c['chunk']):
                    # Carry graph across chunks: this is full BPTT, not detach.
                    a,state=self.fast(self.project(x[:,i:i+c['chunk']].contiguous()),state)
                a=a[:,-1]
            else:a=self.fast(self.project(x))[0][:,-1]
            if s is not None:a=torch.cat([a,self.slow(self.slow_project(s))[0][:,-1]],dim=-1)
            return self.head(a)
    result=dict(case=name,config=c,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),allocator_cap_fraction=.72,synthetic=True,
        caveat='Memory feasibility only. Random normalized inputs, final-origin loss; no trading learning or epoch-time forecast.')
    model=Model().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=c['amp'])
    result['parameters']=sum(p.numel() for p in model.parameters())
    def step(length,slow_length=None):
        x=torch.randn(c['batch'],length,c['dim'],device='cuda')
        s=torch.randn(c['batch'],slow_length,c['slow_dim'],device='cuda') if slow_length else None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=c['amp']):
            loss=model(x,s).float().square().mean()
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize()
        return float(loss.detach())
    try:
        step(min(128,c['length']),min(128,c['slow_length']) if 'slow_length' in c else None)
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        result['baseline_allocated_mib']=torch.cuda.memory_allocated()/2**20
        result['device_free_before_mib']=torch.cuda.mem_get_info()[0]/2**20
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        loss=step(c['length'],c.get('slow_length'))
        result.update(status='PASS',step_seconds=time.perf_counter()-start,loss=loss,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            device_free_after_mib=torch.cuda.mem_get_info()[0]/2**20)
    except torch.OutOfMemoryError as e:
        result.update(status='OOM_AT_72_PERCENT_ALLOCATOR_CAP',error=str(e),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except RuntimeError as e:
        result.update(status='BACKEND_RUNTIME_ERROR',error=str(e),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    Path(out).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('case','status','peak_allocated_mib','peak_reserved_mib') if k in result}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES);p.add_argument('--output',default='reports/context_memory_v1/run_001')
    a=p.parse_args()
    if a.case:worker(a.case,a.output);return
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[]
    for name in CASES:
        path=out/(name+'.json')
        subprocess.run([sys.executable,'-X','utf8',__file__,'--case',name,'--output',str(path)],check=True)
        results.append(json.loads(path.read_text()))
    (out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_MEMORY_PROBE',results=results),indent=2),encoding='utf8')


if __name__=='__main__':main()
