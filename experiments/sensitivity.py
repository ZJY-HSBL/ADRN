"""Bank-only single-seed K and training imbalance sensitivity experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adrn.data import load
from adrn.train import train

tr,va,te,meta=load('bank');res={'dataset':meta,'seed':0,'k':{},'ir':{}}
for k in [4,6,8,10,12]:
 print('K',k,flush=True);_,r,_=train(tr,va,te,seed=0,k=k);res['k'][str(k)]=r['test']
for ir in [10,50,100]:
 print('IR',ir,flush=True);ir_train,ir_val,ir_test,_=load('bank',target_ir=ir,ir_seed=0);_,r,_=train(ir_train,ir_val,ir_test,seed=0,k=8);y=ir_train[1];n=int((y==1).sum());neg=int((y==0).sum());res['ir'][str(ir)]={'test':r['test'],'train_negative':neg,'train_positive':n,'actual_ratio':neg/n}
Path('results/sensitivity.json').write_text(json.dumps(res,indent=2))
for section in ('k','ir'):
 data=res[section];fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
 for metric in ('f1','f2','pr_auc'):
  vals=[(v['test'] if section=='ir' else v)[metric] for v in data.values()];ax.plot(list(map(int,data)),vals,marker='o',label=metric.upper())
 ax.set_xlabel('Number of bases per class (K)' if section=='k' else 'Target training imbalance ratio');ax.set_ylabel('Test metric');ax.set_title('BANK · '+('basis count' if section=='k' else 'imbalance pressure')+' · seed 0',loc='left',weight='bold');ax.grid(alpha=.25);ax.legend(frameon=False);fig.savefig(f'figures/bank_{section}_sensitivity.png',dpi=150);plt.close(fig)
