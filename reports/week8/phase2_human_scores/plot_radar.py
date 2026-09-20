"""Plot the frozen Week8 anonymous human dimension means."""
from pathlib import Path
import csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]
rows=list(csv.DictReader((root/'reports/week8/phase2_human_scores/summary.csv').open()))
dims=['accuracy','completeness','logic','safety','format'];angles=[i*2*math.pi/5 for i in range(5)];angles+=[angles[0]]
fig,ax=plt.subplots(figsize=(8,6),subplot_kw={'polar':True})
labels={'original_base':'Original base','final_sft':'SFT','final_dpo':'DPO'}
for r in sorted(rows,key=lambda r:float(r['weighted_total']),reverse=True):
 values=[float(r[d]) for d in dims];ax.plot(angles,values+[values[0]],label=labels[r['model_id']],linewidth=2)
ax.set_xticks(angles[:-1],['Accuracy','Completeness','Logic','Safety','Format']);ax.set_ylim(0,5);ax.set_yticks([1,2,3,4,5]);ax.set_title('Week 8 — Anonymous human scores\n20 questions, two reviewers | 2026-09-18',pad=25);ax.legend(loc='upper left',bbox_to_anchor=(1.1,1.1));fig.tight_layout();fig.savefig(root/'reports/figures/week8_human_radar.png',dpi=180);plt.close(fig)
