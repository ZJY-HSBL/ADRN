"""Default alternative category mapping: explicit hypothesis, not author protocol."""
import json
from pathlib import Path
import numpy as np
from adrn.data import load
from adrn.train import train

def main():
    tr,va,te,meta=load('default',default_encoding='grouped')
    runs=[]
    for seed in range(5):
        print('grouped encoding seed',seed,flush=True)
        _,r,_=train(tr,va,te,seed=seed,epochs=6,pretrain_epochs=3);runs.append(r)
    summary={m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1))} for m in runs[0]['test']}
    output={'data':meta,'mapping':{'EDUCATION':{'0':4,'5':4,'6':4},'MARRIAGE':{'0':3}},'config':{'seeds':[0,1,2,3,4],'epochs':6,'pretrain_epochs':3,'hidden':64,'latent':16,'k':4,'batch_size':512,'lr':.001,'beta':.5},'summary':summary,'runs':runs}
    Path('results/default_grouped_encoding.json').write_text(json.dumps(output,indent=2))
if __name__=='__main__':main()
