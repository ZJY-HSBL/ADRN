"""Inspect the first fixed-configuration run without refitting or selecting on test."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adrn.data import load
from adrn.model import ADRN

def main():
    summary={}
    for name in ['gmsc','bank','default']:
        _,_,(x,y),_=load(name);ckpt=torch.load(f'checkpoints/{name}_seed0.pt',map_location='cpu',weights_only=True)
        model=ADRN(ckpt['input_dim'],ckpt['hidden'],ckpt['latent'],ckpt['k']);model.load_state_dict(ckpt['state_dict']);model.eval()
        with torch.no_grad():d={k:v.numpy() for k,v in model(torch.from_numpy(x),return_details=True).items()}
        before=d['observed'];after=d['probability'];struct=d['structure'];gate=d['gate'];delta=after-before
        summary[name]={'test_count':len(y),'positive_count':int(y.sum()),'positive_deficit_count':int(np.count_nonzero(d['deficit']>0)),'positive_class_deficit_count':int(np.count_nonzero((d['deficit']>0)&(y==1))),'mean_overlap':float(np.mean(d['overlap'])),'mean_gate':float(np.mean(gate)),'mean_correction':float(np.mean(delta))}
        idx=np.random.default_rng(7).choice(len(y),min(2500,len(y)),replace=False)
        fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
        ax[0].scatter(before[idx],struct[idx],c=y[idx],s=4,alpha=.32,cmap='coolwarm');ax[0].plot([0,1],[0,1],ls='--',c='gray');ax[0].set(xlabel='Observed probability',ylabel='Structural probability',title='Structure vs observation')
        ax[1].hist(delta,bins=40,color='#6253c7');ax[1].set(xlabel='Probability correction',ylabel='Count',title='Applied correction')
        ax[2].scatter(d['overlap'][idx],gate[idx],s=4,alpha=.2,color='#3d9a9b');ax[2].set(xlabel='Local overlap',ylabel='Gate',title='Overlap and gate')
        fig.suptitle(name.upper()+' · model mechanism on test data · seed 0',weight='bold');fig.savefig('figures/'+name+'_mechanism.png',dpi=150);plt.close(fig)
    Path('results/mechanism.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
