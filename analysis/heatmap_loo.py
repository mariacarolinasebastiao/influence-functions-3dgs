import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

from configs.paths import PROJ_ROOT, WORK_ROOT
RES       = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/room_splatfacto/results_fim_all_frames")
LOO_DIR   = Path(f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto")
INF_GROUP = "all_groups"
TESTS     = [28]

ov_data   = torch.load(RES / "overlap_vs_influence_data.pt")
pose_data = torch.load(RES / "pose_influence_data.pt")
overlap   = ov_data["overlap_matrix"].numpy()       # (n_test, n_train)
trans     = pose_data["trans_dist_matrix"].numpy()  # (n_train, n_test)
rot       = pose_data["rot_dist_matrix"].numpy()    # (n_train, n_test)

all_deltas = {}
for t in TESTS:
    d = torch.load(LOO_DIR / f"loo_clean_abs" / f"loo_deltas_clean_test{t}.pt")
    for k, dvec in d.items():
        if k not in all_deltas:
            all_deltas[k] = dvec

all_train = sorted(all_deltas.keys())
n_cols    = len(all_train)
n_rows    = len(TESTS)

loo_mat   = np.full((n_rows, n_cols), np.nan)
ov_mat    = np.full((n_rows, n_cols), np.nan)
trans_mat = np.full((n_rows, n_cols), np.nan)
rot_mat   = np.full((n_rows, n_cols), np.nan)
inf_mat   = np.full((n_rows, n_cols), np.nan)
dot_mat   = np.full((n_rows, n_cols), np.nan)

for i, t in enumerate(TESTS):
    inf_full = torch.load(RES / INF_GROUP / f"test_image_{t}.pt").squeeze().numpy()
    dot_full = torch.load(RES / INF_GROUP / f"dot_test_image_{t}.pt").squeeze().numpy()
    for j, k in enumerate(all_train):
        loo_mat[i, j]   = all_deltas[k][t] if t in all_deltas[k] else np.nan
        ov_mat[i, j]    = overlap[t, k]
        trans_mat[i, j] = trans[k, t]
        rot_mat[i, j]   = rot[k, t]
        inf_mat[i, j]   = inf_full[k]
        dot_mat[i, j]   = dot_full[k]


def corr_str(metric_mat, loo_m):
    """
    Compute per-row (test view) Pearson and Spearman correlation between
    metric_mat and loo_m using raw (unnormalised) values.
    Returns a list of annotation strings, one per row.
    """
    strs = []
    for i in range(metric_mat.shape[0]):
        m   = metric_mat[i]
        lo  = loo_m[i]
        ok  = ~np.isnan(m) & ~np.isnan(lo)
        if ok.sum() < 3:
            strs.append("n/a")
            continue
        rp  = pearsonr(m[ok],  lo[ok])[0]
        rs  = spearmanr(m[ok], lo[ok])[0]
        strs.append(f"$r_P={rp:+.2f}$, $r_S={rs:+.2f}$")
    return strs


# Compute correlations on raw matrices BEFORE sorting
corrs = {
    "inf":   corr_str(inf_mat,   loo_mat),
    "dot":   corr_str(dot_mat,   loo_mat),
    "ov":    corr_str(ov_mat,    loo_mat),
    "trans": corr_str(trans_mat, loo_mat),
    "rot":   corr_str(rot_mat,   loo_mat),
}


def make_title(base, corr_list):
    """Append correlation(s) to panel title; one line per test view if multiple."""
    if len(corr_list) == 1:
        return f"{base}   {corr_list[0]}"
    # multiple rows: show each on its own line
    lines = [base] + [f"  Test {t}: {c}" for t, c in zip(TESTS, corr_list)]
    return "\n".join(lines)


# Sort columns by LOO delta (most harmful → most beneficial)
col_order        = np.argsort(np.nanmean(loo_mat, axis=0))
loo_mat          = loo_mat[:, col_order]
ov_mat           = ov_mat[:, col_order]
trans_mat        = trans_mat[:, col_order]
rot_mat          = rot_mat[:, col_order]
inf_mat          = inf_mat[:, col_order]
dot_mat          = dot_mat[:, col_order]
all_train_sorted = [all_train[k] for k in col_order]


def normalize_by_max_abs_value(mat):
    rmax = np.nanmax(np.abs(mat), axis=1, keepdims=True)
    rmax = np.where(rmax == 0, 1, rmax)
    return mat / rmax

def normalize_by_range(mat, higher_is_green=True):
    rmin   = np.nanmin(mat, axis=1, keepdims=True)
    rmax   = np.nanmax(mat, axis=1, keepdims=True)
    scaled = (mat - rmin) / (rmax - rmin + 1e-12)
    return (2 * scaled - 1) if higher_is_green else (1 - 2 * scaled)


cmap = LinearSegmentedColormap.from_list("diverging", ["#bd310e", "white", "#0ca120"])

panels = [
    (normalize_by_max_abs_value(loo_mat),
     r"LOO $\Delta\mathcal{L}$  (ground truth)"),
    (normalize_by_max_abs_value(inf_mat),
     make_title(f"eFIM influence ({INF_GROUP})", corrs["inf"])),
    (normalize_by_max_abs_value(dot_mat),
     make_title(f"Gradient dot product ({INF_GROUP})", corrs["dot"])),
    (normalize_by_range(ov_mat,    higher_is_green=True),
     make_title("Gaussian overlap (higher = green)", corrs["ov"])),
    (normalize_by_range(trans_mat, higher_is_green=False),
     make_title("Translation distance (closer = green)", corrs["trans"])),
    (normalize_by_range(rot_mat,   higher_is_green=False),
     make_title("Rotation distance (closer = green)", corrs["rot"])),
]

with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13}):
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True, sharey=True)

    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1,interpolation="nearest")
        ax.set_ylabel("Test view")
        ax.set_title(title)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"Test {t}" for t in TESTS])
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    axes[-1].set_xlabel(r"Training image $k$ (sorted by LOO $\Delta\mathcal{L}$)")
    step = max(1, n_cols // 30)
    axes[-1].set_xticks(range(0, n_cols, step))
    axes[-1].set_xticklabels([str(all_train_sorted[i]) for i in range(0, n_cols, step)], rotation=90, fontsize=8)

    plt.tight_layout()
    out = RES / f"all_metrics_heatmap_test{'_'.join(map(str, TESTS))}_abs_all.jpg"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", out)
