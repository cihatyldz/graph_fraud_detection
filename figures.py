"""Sekil 1: rolling-origin, guclu baseline uzerinde origin bazinda PR-AUC farki (concatenation vs router).
Veriler safeg_numbers'tan; baski icin vektorel PDF + inceleme icin PNG."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import safeg_numbers as N

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})

ie, el = N.D['ieee'], N.D['ell']
ieee_x = [f"q={float(r[1]):.1f}\nκ={float(r[2]):.2f}" for r in ie['p4_origin_pr']]
ieee_cat = [float(r[6]) for r in ie['p4_origin_pr']]
ieee_rou = [float(r[8]) for r in ie['p4_origin_pr']]
e4 = el['e4_origin_pr']
ell_x = [f"t≤{i.split()[1]}\nκ={e4.loc[i, 'kappa']:.2f}" for i in e4.index]
ell_cat, ell_rou = list(e4['D-AA']), list(e4['RC-AA'])

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
for ax, title, xs, cat, rou in [(axes[0], 'IEEE-CIS (base: A$_{full}$; LightGBM, XGBoost)', ieee_x, ieee_cat, ieee_rou),
                                (axes[1], 'Elliptic++ (base: A+AGG; LR, RF, XGBoost, LightGBM)', ell_x, ell_cat, ell_rou)]:
    x = np.arange(len(xs)); w = 0.36
    ax.bar(x - w / 2 - 0.01, cat, w, color=BLUE, edgecolor='white', linewidth=1.0, label='Concatenation (base + graph)', zorder=3)
    ax.bar(x + w / 2 + 0.01, rou, w, color=ORANGE, edgecolor='white', linewidth=1.0, label='Router (seen → base + graph, else base)', zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=7)
    ax.set_title(title, fontsize=8, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axes[0].set_ylabel('Δ PR-AUC vs. strong baseline')
axes[1].set_xlabel('Rolling origin (last training time step) and entity coverage', fontsize=7, color=MUTED)
axes[0].set_xlabel('Rolling origin (training-window quantile) and entity coverage', fontsize=7, color=MUTED)
fig.legend(handles=[Patch(color=BLUE, label='Concatenation (base + graph)'), Patch(color=ORANGE, label='Router (seen entity → base + behavioural + graph; otherwise base)')],
           loc='upper center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_rolling_origin')
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
print('yazildi', out)
