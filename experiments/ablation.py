"""Paper's four structural ablations; same split and training config."""
import argparse,json
from pathlib import Path
from adrn.data import load
from adrn.train import train

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['gmsc','bank','default']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=6);p.add_argument('--pretrain-epochs',type=int,default=3);p.add_argument('--resume',action='store_true');args=p.parse_args()
    for name in args.datasets:
        tr,va,te,_=load(name);path=Path('results',name+'_ablation.json');out=json.loads(path.read_text()) if args.resume and path.exists() else {}
        for variant in ['no_reliability','signed_deficit','no_gate','no_latent']:
            out.setdefault(variant,[])
            for seed in args.seeds:
                if any(r['seed']==seed for r in out[variant]):continue
                print(name,variant,seed,flush=True)
                _,r,_=train(tr,va,te,seed=seed,epochs=args.epochs,pretrain_epochs=args.pretrain_epochs,variant=variant)
                out[variant].append(r);path.write_text(json.dumps(out,indent=2));print(json.dumps(r['test']),flush=True)
    combined={}
    for name in ('gmsc','bank','default'):
        path=Path('results',name+'_ablation.json')
        if path.exists():combined[name]=json.loads(path.read_text())
    Path('results/ablation.json').write_text(json.dumps(combined,indent=2))
if __name__=='__main__':main()
