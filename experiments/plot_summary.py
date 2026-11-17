"""Generate figures only from stored experiment results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

s=json.loads(Path('results/summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for ax,key in zip(axes,['f1','f2','pr_auc']):
    vals=[s[n]['summary'][key]['mean'] for n in s]; sd=[s[n]['summary'][key]['sd'] or 0 for n in s]
    ax.bar(list(s),vals,yerr=sd,capsize=4,color=['#6c5ce7','#3d9a9b','#e6a148']);ax.set_title(key.upper());ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
fig.suptitle('ADRN · held-out test results (mean ± sample SD, 5 seeds)',weight='bold');fig.savefig('figures/test_metrics.png',dpi=160);plt.close(fig)
if Path('results/ablation.json').exists():
    a=json.loads(Path('results/ablation.json').read_text());metrics=['f1','f2','pr_auc'];labels=['Full','No reliability','Signed deficit','No gate','No latent'];variants=['no_reliability','signed_deficit','no_gate','no_latent']
    for name in a:
        full=next(r for r in s[name]['runs'] if r['seed']==0)['test'];values=[full]+[a[name][v][0]['test'] for v in variants]
        x=np.arange(len(labels));fig,ax=plt.subplots(figsize=(9.5,4),constrained_layout=True)
        for i,m in enumerate(metrics):ax.bar(x+(i-1)*.23,[v[m] for v in values],width=.23,label=m.upper())
        ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.set_title(name.upper()+' · ablations · seed 0 only',loc='left',weight='bold');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,ncol=3)
        fig.savefig('figures/'+name+'_ablation.png',dpi=150);plt.close(fig)
