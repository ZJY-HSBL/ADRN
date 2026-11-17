"""Paper's four structural ablations; same split and training config."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adrn.data import load
from adrn.train import train

p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--epochs',type=int,default=6);p.add_argument('--pretrain-epochs',type=int,default=3);args=p.parse_args()
out={}
for name in args.datasets:
    tr,va,te,meta=load(name);out[name]={}
    for variant in ['no_reliability','signed_deficit','no_gate','no_latent']:
        out[name][variant]=[]
        for seed in args.seeds:
            print(name,variant,seed,flush=True)
            _,r,_=train(tr,va,te,seed=seed,epochs=args.epochs,pretrain_epochs=args.pretrain_epochs,variant=variant)
            out[name][variant].append(r);print(json.dumps(r['test']),flush=True)
    Path('results',name+'_ablation.json').write_text(json.dumps(out[name],indent=2))
all_results={}
for name in ('gmsc','bank','default'):
    path=Path('results',name+'_ablation.json')
    if path.exists(): all_results[name]=json.loads(path.read_text())
Path('results/ablation.json').write_text(json.dumps(all_results,indent=2))
