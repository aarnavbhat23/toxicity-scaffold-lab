import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data=json.loads(Path('results/metrics.json').read_text())['models'];tasks=list(data['morgan_logistic']['assays'])
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(len(tasks))
for name,label,color in [('morgan_logistic','Morgan logistic','#ad661b'),('message_passing_ensemble','Graph ensemble','#216e91')]:
    axes[0].plot(x,[data[name]['assays'][t]['auroc'] for t in tasks],'o-',label=label,color=color)
    axes[1].plot(x,[data[name]['assays'][t]['sets']['row_calibration']['scaffold_simultaneous_coverage'] for t in tasks],'o-',label=label+' / row calibration',color=color)
    axes[1].plot(x,[data[name]['assays'][t]['sets']['scaffold_max_calibration']['scaffold_simultaneous_coverage'] for t in tasks],'o--',label=label+' / scaffold calibration',color=color,alpha=.7)
for ax in axes:ax.set_xticks(x,tasks,rotation=65,ha='right');ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Held-out scaffold AUROC');axes[0].legend(frameon=False)
axes[1].set_ylabel('Simultaneous scaffold coverage');axes[1].axhline(.9,color='#444',ls=':');axes[1].set_ylim(0,1.05)
axes[1].legend(fontsize=8,frameon=False,loc='lower left')
fig.savefig('results/assays.png',dpi=180);fig.savefig('results/assays.svg')
