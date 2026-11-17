"""Five-seed ablation summaries and figures; only stored runs are used."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

full=json.loads(Path('results/summary.json').read_text());result={}
for name in ('gmsc','bank','default'):
    ab=json.loads(Path('results',name+'_ablation.json').read_text());result[name]={}
    variants={'full':full[name]['runs'],**ab}
    for variant,runs in variants.items():
        result[name][variant]={'n':len(runs),'metrics':{m:{'mean':float(np.mean([r['test'][m] for r in runs])),'sd':float(np.std([r['test'][m] for r in runs],ddof=1)) if len(runs)>1 else None} for m in ['f1','f2','pr_auc']}}
    x=np.arange(len(variants));fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for i,m in enumerate(['f1','f2','pr_auc']):
        vals=[result[name][v]['metrics'][m]['mean'] for v in variants];err=[result[name][v]['metrics'][m]['sd'] or 0 for v in variants]
        ax.bar(x+(i-1)*.23,vals,.23,yerr=err,capsize=2,label=m.upper())
    ax.set_xticks(x,['Full','No reliability','Signed deficit','No gate','No latent']);ax.set_ylim(0,1);ax.set_title(name.upper()+' · ablations · mean ± sample SD',loc='left',weight='bold');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,ncol=3);fig.savefig(f'figures/{name}_ablation_means.png',dpi=150);plt.close(fig)
Path('results/ablation_means.json').write_text(json.dumps(result,indent=2))
