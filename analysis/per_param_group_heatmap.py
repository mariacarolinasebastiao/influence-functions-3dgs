import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from configs.paths import WORK_ROOT
RES           = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
LOO_DELTA_DIR = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto")
OUT_DIR       = RES

TESTS = [14, 16]   # one output file per test view

PARAM_GROUPS = [
    ("means",              "means"),
    ("means_scales",       "m+s"),
    ("means_scales_quats", "m+s+q"),
    ("colors",             "colors"),
    ("five_params",        "5p"),
    ("all_groups",         "all-6"),
]

cmap = LinearSegmentedColormap.from_list("diverging", ["#bd310e", "white", "#0ca120"])


def normalize_by_max_abs(vec):
    """Row-wise normalisation by max |value|; safe for all-NaN rows."""
    rmax = np.nanmax(np.abs(vec))
    return vec / rmax if rmax > 0 else vec


def load_loo_deltas(t):
    """Load delta file for test view t. Returns dict {removed_idx: delta_float} or {}."""
    candidates = [
        LOO_DELTA_DIR / f"loo_top10_test{t}" / f"loo_deltas_top10_test{t}.pt",
    ]
    for p in candidates:
        if p.exists():
            raw = torch.load(p)
            out = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    if t in v:
                        out[k] = float(v[t])
                else:
                    out[k] = float(v)
            return out
    print(f"  WARNING: no LOO delta file found for test {t}")
    return {}


for t in TESTS:
    print(f"\n=== Test view {t} ===")

    deltas = load_loo_deltas(t)
    if not deltas:
        continue

    all_train = sorted(deltas.keys())
    n_cols    = len(all_train)
    print(f"  {n_cols} removed-image LOO entries")

    # --- LOO vector ---
    loo_vec = np.array([deltas[k] for k in all_train])

    # --- influence vectors per parameter group ---
    inf_vecs = {}
    for folder, label in PARAM_GROUPS:
        inf_path = RES / folder / f"test_image_{t}.pt"
        if inf_path.exists():
            inf_full = torch.load(inf_path).squeeze().numpy()
            inf_vecs[folder] = np.array([inf_full[k] if k < len(inf_full) else np.nan for k in all_train])
        else:
            print(f"  skip {folder}: {inf_path} not found")

    # --- sort columns by LOO delta (ascending: most harmful → most beneficial) ---
    col_order   = np.argsort(loo_vec)
    loo_sorted  = loo_vec[col_order]
    train_sorted = [all_train[i] for i in col_order]

    # --- build panels ---
    panels = [(normalize_by_max_abs(loo_sorted), r"LOO $\Delta\mathcal{L}$ (ground truth)")]
    for folder, label in PARAM_GROUPS:
        if folder in inf_vecs:
            inf_sorted = inf_vecs[folder][col_order]
            panels.append((normalize_by_max_abs(inf_sorted), f"eFIM influence ({label})"))

    n_panels = len(panels)

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13}):
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        for ax, (vec, title) in zip(axes, panels):
            im = ax.imshow(vec[np.newaxis, :], aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(title)
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

        axes[-1].set_xlabel(r"Training image $k$ (sorted by LOO $\Delta\mathcal{L}$, harmful → beneficial)")
        step = max(1, n_cols // 30)
        axes[-1].set_xticks(range(0, n_cols, step))
        axes[-1].set_xticklabels([str(train_sorted[i]) for i in range(0, n_cols, step)], rotation=90, fontsize=8)

        fig.suptitle(f"Garden - test view {t}: parameter-group influence vs LOO",fontsize=14, y=1.01)
        plt.tight_layout()
        out = OUT_DIR / f"param_group_heatmap_test{t}.jpg"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")
