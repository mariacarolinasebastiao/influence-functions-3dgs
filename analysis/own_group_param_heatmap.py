import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

from configs.paths import PROJ_ROOT, WORK_ROOT
RES     = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
OUT_DIR = RES
DELTA_ROOTS = [
    Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto"),
    Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto"),
    Path(f"{WORK_ROOT}/nerfstudio/nerfstudio/outputs"),
]
TESTS        = [16, 14]
PARAM_GROUPS = [("means","means"),("means_scales","m+s"),("means_scales_quats","m+s+q"),("colors","colors"),("five_params","5p"),("all_groups","all-6")]
TOP_K, MIN_COVERAGE = 5, 6
MODE = "abs"   # "signed" | "abs"

cmap = LinearSegmentedColormap.from_list("diverging", ["#bd310e", "white", "#0ca120"])
cmap.set_bad(color="#cccccc")

def norm_max_abs(vec):
    m = np.nanmax(np.abs(vec)); return vec / m if m > 0 else vec

def load_all_loo_pool(test_view):
    deltas_by_k = {}
    for r in DELTA_ROOTS:
        if not r.exists(): continue
        for p in r.rglob("loo_deltas_*test*.pt"):
            try: d = torch.load(p)
            except Exception as e: print(f"  skip {p}: {e}"); continue
            for k, vec in d.items():
                if k not in deltas_by_k: deltas_by_k[k] = vec
    return {k: float(vec[test_view]) for k, vec in deltas_by_k.items()
            if isinstance(vec, dict) and test_view in vec}

for t in TESTS:
    print(f"\n=== Test view {t} ===")
    loo_pool = load_all_loo_pool(t)
    print(f"  LOO pool: {len(loo_pool)} indices")
    panels = []

    for folder, label in PARAM_GROUPS:
        inf_path = RES / folder / f"test_image_{t}.pt"
        if not inf_path.exists(): print(f"  skip {label}: missing"); continue

        inf_full = torch.load(inf_path).squeeze().numpy()
        ranked   = np.argsort(inf_full)
        selected = list(ranked[:TOP_K]) + list(ranked[-TOP_K:][::-1]) if MODE == "signed" \
                   else list(np.argsort(np.abs(inf_full))[-TOP_K*2:])

        covered = [k for k in selected if k in loo_pool]
        print(f"  {label}: coverage {len(covered)}/{len(selected)}", end="")
        if len(covered) < MIN_COVERAGE: print(f"  → skip"); continue
        print()

        inf_vec = np.array([inf_full[k] for k in selected], dtype=float)
        loo_vec = np.array([loo_pool.get(k, np.nan) for k in selected], dtype=float)
        x = np.array([inf_full[k] for k in covered])
        y = np.array([loo_pool[k]  for k in covered])
        rp, rs = pearsonr(x, y)[0], spearmanr(x, y)[0]

        order = np.argsort(loo_vec)
        panels.append((norm_max_abs(inf_vec[order]), norm_max_abs(loo_vec[order]),
                       [selected[i] for i in order], label, rp, rs))

    if not panels: print("  No groups with sufficient coverage — skipping."); continue

    n_panels = len(panels)
    with plt.rc_context({"font.family":"serif","mathtext.fontset":"cm","font.size":11,"axes.labelsize":11,"axes.titlesize":12}):
        fig, axes = plt.subplots(n_panels, 2, figsize=(10, 2.0 + 1.8*n_panels), gridspec_kw={"width_ratios":[1,1]})
        if n_panels == 1: axes = axes[np.newaxis, :]

        for row, (inf_s, loo_s, idx_s, label, rp, rs) in enumerate(panels):
            n_cols = len(idx_s); step = max(1, n_cols // 10)
            xticks = range(0, n_cols, step); xlabels = [str(idx_s[i]) for i in xticks]

            ax = axes[row, 0]
            im = ax.imshow(np.ma.masked_invalid(inf_s[np.newaxis,:]), aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"eFIM influence ({label})   $r_P={rp:+.2f}$,  $r_S={rs:+.2f}$", fontsize=11)
            ax.set_yticks([]); ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02)

            ax2 = axes[row, 1]
            im2 = ax2.imshow(np.ma.masked_invalid(loo_s[np.newaxis,:]), aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax2.set_title(r"LOO $\Delta\mathcal{L}$ (ground truth)", fontsize=11)
            ax2.set_yticks([]); ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, rotation=90, fontsize=8)
            plt.colorbar(im2, ax=ax2, fraction=0.08, pad=0.02)

        fig.suptitle(f"Garden - test view {t}: each group on its own top/bottom-{TOP_K} selection\n(columns sorted by LOO $\\Delta\\mathcal{{L}}$, harmful → beneficial)", fontsize=12, y=1.01)
        plt.tight_layout()
        out = OUT_DIR / f"param_group_ownset_heatmap_test{t}_abs.jpg"
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"  Saved {out}")
