"""Visualize NDU score-segment removal rates from stored run results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
for ax,name in zip(axes,['gmsc','bank','default']):
    r=json.load(open('results/'+name+'_ndu_approx.json'));groups=r['runs'][0]['segments'];rates=[v['rate']*100 for v in groups]
    ax.bar(range(1,len(groups)+1),rates,color='#665acb');ax.set_xticks(range(1,len(groups)+1));ax.set_xlabel('F-score segment (low → high)');ax.set_ylabel('Removal rate (%)');ax.set_title(name.upper()+' · seed 0');ax.grid(axis='y',alpha=.2)
fig.suptitle('Approximate NDU · removal by F-score segment',weight='bold');fig.savefig('figures/ndu_segment_rates.png',dpi=150);plt.close(fig)
