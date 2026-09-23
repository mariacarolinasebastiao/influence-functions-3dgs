import torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

from configs.paths import PROJ_ROOT, WORK_ROOT
GARDEN_INF  = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
GARDEN_PROJ = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto")
BICY_INF    = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/results_fim_zerofilter")
BICY_PROJ   = Path(f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto")
ROOM_INF    = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/room_splatfacto/results_fim_all_frames")
ROOM_PROJ   = Path(f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto")

def gclean(t): return GARDEN_PROJ / "loo_clean_abs" / f"loo_deltas_clean_test{t}.pt"

SCENES = [
    ("Garden 3",  GARDEN_INF, 3,  gclean(3)),
    ("Garden 9",  GARDEN_INF, 9,  gclean(9)),
    ("Garden 13", GARDEN_INF, 13, gclean(13)),
    ("Garden 14", GARDEN_INF, 14, gclean(14)),
    ("Garden 16", GARDEN_INF, 16, gclean(16)),
    ("Garden 17", GARDEN_INF, 17, gclean(17)),
    #("Bicycle 7", BICY_INF,   7,  BICY_PROJ / "loo_clean_abs" / "loo_deltas_clean_test7.pt"),
    ("Room 28",   ROOM_INF,   28, ROOM_PROJ / "loo_clean_abs" / "loo_deltas_clean_test28.pt"),
]

def load_means(inf_dir, t):
    d = torch.load(inf_dir / "colors" / f"test_image_{t}.pt", map_location="cpu")
    return d.float().squeeze().numpy() if isinstance(d, torch.Tensor) \
        else np.array([float(d[k]) for k in sorted(d.keys())])

views = [s[0] for s in SCENES]
metrics = {'Influence (colors)': [], 'Gaussian overlap': [],
           r'$-$trans.\ dist.': [], r'$-$rot.\ dist.': []}

for name, inf_dir, t, clean_path in SCENES:
    inf  = load_means(inf_dir, t)
    loo  = torch.load(clean_path, map_location="cpu")        # {idx: {test: delta}}
    idxs = sorted(loo.keys())                                 # |influence| top-10
    y    = np.array([loo[k][t] for k in idxs])

    pose = torch.load(inf_dir / "pose_influence_data.pt", map_location="cpu")
    ovd  = torch.load(inf_dir / "overlap_vs_influence_data.pt", map_location="cpu")
    trans = np.array(pose["trans_dist_matrix"]); rot = np.array(pose["rot_dist_matrix"])
    ov    = ovd["overlap_matrix"].numpy()

    r_inf = spearmanr([inf[k]       for k in idxs], y)[0]
    r_ov  = spearmanr([ov[t, k]     for k in idxs], y)[0]
    r_tr  = spearmanr([-trans[k, t] for k in idxs], y)[0]
    r_ro  = spearmanr([-rot[k, t]   for k in idxs], y)[0]
    metrics['Influence (colors)'].append(r_inf)
    metrics['Gaussian overlap' ].append(r_ov)
    metrics[r'$-$trans.\ dist.'].append(r_tr)
    metrics[r'$-$rot.\ dist.'  ].append(r_ro)
    print(f"{name:<11} inf={r_inf:+.3f}  ov={r_ov:+.3f}  tr={r_tr:+.3f}  ro={r_ro:+.3f}")


colors = ['#6E9CC5', '#F4A261', '#8DC18F', '#E76F51']
n_views, n_metrics = len(views), len(metrics)
bar_width = 0.2
x = np.arange(n_views)
offsets = np.linspace(-bar_width * 1.5, bar_width * 1.5, n_metrics)

with plt.rc_context({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": True, "axes.spines.right": True,
}):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ((label, values), color) in enumerate(zip(metrics.items(), colors)):
        ax.bar(x + offsets[i], values, bar_width, color=color, edgecolor='none', label=label)
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=25, ha='right', rotation_mode='anchor')
    ax.set_ylabel(r"$r_S$ with LOO $\Delta\mathcal{L}$")
    ax.legend(loc='upper left', framealpha=0.95, ncol=2, fontsize=11)
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.set_ylim(-0.5, 1.05)
    plt.tight_layout()
    out = Path(f"{WORK_ROOT}/nerfstudio/nerfstudio/alternatives_vs_loo_bars_colors_abs.jpg")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
print(f"Saved: {out}")
