"""Welch tests with separate BH families for baselines and ablations."""
import json
from pathlib import Path
from adrn.statistics import bh,welch

p=Path('results');full=json.loads((p/'summary.json').read_text());families={}
for family in ['baselines','ablation']:
    comparisons=[]
    for name in ('gmsc','bank','default'):
        other=json.loads((p/(name+'_'+family+'.json')).read_text())
        variants=other['methods'] if family=='baselines' else other
        for variant,details in variants.items():
            runs=details['runs'] if family=='baselines' else details
            if len(runs)<2:continue
            for metric in ('f1','f2','pr_auc'):
                a=[r['test'][metric] for r in full[name]['runs']];b=[r['test'][metric] for r in runs]
                comparisons.append({'dataset':name,'variant':variant,'metric':metric,'raw_p':welch(a,b),'n_full':len(a),'n_other':len(b)})
    if comparisons:
        for x,q in zip(comparisons,bh([x['raw_p'] for x in comparisons])):x['bh_p']=float(q)
    families[family]=comparisons
(p/'significance.json').write_text(json.dumps(families,indent=2))
print({k:len(v) for k,v in families.items()})
