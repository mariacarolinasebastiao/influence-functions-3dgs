import torch, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": True, "axes.spines.right": True,
})

from configs.paths import PROJ_ROOT, WORK_ROOT
RES    = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
LOODIR = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto")
tests  = [3, 9, 13, 14,16, 17]

cmap = LinearSegmentedColormap.from_list("diverging", ["#bd310e", "white", "#0ca120"])

# ---- load ----
DELTA_ROOTS = [LOODIR,Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto"),]
deltas_by_k = {}
for r in DELTA_ROOTS:
    if not r.exists(): continue
    for p in r.rglob("loo_deltas_*test*.pt"):
        try: d = torch.load(p)
        except: continue
        for k, vec in d.items():
            if k not in deltas_by_k: deltas_by_k[k] = vec

GROUP = "means"
K = 5
data = {}
for t in tests:
    inf = torch.load(RES / GROUP / f"test_image_{t}.pt").squeeze().numpy()
    wanted = list(np.argsort(np.abs(inf))[-K*2:])
    ks = [k for k in wanted if k in deltas_by_k and t in deltas_by_k[k]]
    if len(ks) < 2: continue
    infv = np.array([inf[k] for k in ks]) / 1e3
    loov = np.array([deltas_by_k[k][t] for k in ks])
    vmax = np.abs(infv).max()
    data[t] = dict(inf=infv, loo=loov, inf_norm=infv/(vmax+1e-12), rp=pearsonr(infv,loov)[0], rs=spearmanr(infv,loov)[0])
    
# ---- 2 rows × 3 cols ----
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
axes = axes.flatten()

for ax, t in zip(axes, sorted(tests)):
    D = data[t]
    sc = ax.scatter(D['inf'], D['loo'] * 1e3, c=D['inf_norm'], cmap=cmap, vmin=-1, vmax=1, s=95, alpha=.9, edgecolors="none", zorder=3)
    ax.set_facecolor("#e1e9f6")
    ax.axhline(0, color="0.6", lw=.9, ls="--")
    ax.axvline(0, color="0.6", lw=.9, ls="--")
    ax.set_title(rf"Test {t}   $r_P$={D['rp']:+.2f}   $r_S$={D['rs']:+.2f}")
    ax.set_xlabel(r"Influence score (means, $\times 10^{3}$)")
    ax.set_ylabel(r"LOO $\Delta\mathcal{L}$  ($\times 10^{3}$)")
    ax.grid(alpha=.25, lw=.6)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)

plt.tight_layout()
out = RES/"influence_vs_loo_garden.jpg"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)

