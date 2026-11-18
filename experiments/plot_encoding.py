"""Compare the two explicit Default encodings using saved experiment results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results = Path('results')
raw = json.loads((results / 'summary.json').read_text())['default']['summary']
grouped = json.loads((results / 'default_grouped_encoding.json').read_text())['summary']
metrics = ['f1', 'f2', 'pr_auc']
x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for i, (label, summary) in enumerate([('Raw categories (33)', raw), ('Grouped categories (29)', grouped)]):
    ax.bar(x + (i - .5) * .32, [summary[m]['mean'] for m in metrics], .32,
           yerr=[summary[m]['sd'] for m in metrics], capsize=3, label=label)
ax.set_xticks(x, ['F1', 'F2', 'PR-AUC'])
ax.set_ylim(0, 1)
ax.set_title('Default encoding comparison · mean ± sample SD', loc='left', weight='bold')
ax.grid(axis='y', alpha=.2)
ax.legend(frameon=False)
fig.savefig('figures/default_encoding_comparison.png', dpi=150)
