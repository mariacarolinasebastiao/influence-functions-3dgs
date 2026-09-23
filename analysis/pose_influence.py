import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from pytorch_msssim import ssim
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from gsplat.cuda._wrapper import fully_fused_projection
from src.fim_influence import (
    FIMInfluenceModule, GSObjective, CustomGSDataset, custom_collate_fn,
    camera_distances, load_influence, compute_influences_for_groups,
    generate_influence_files, project_gaussians_for_camera, contributing_gaussians,
    ABLATION_GROUPS, GROUP_FOLDERS, INFLUENCE_PARAMS, DEVICE,
)
from configs.scenes import build_scene


def compute_pose_influence_correlations(pipeline, results_dir, groups_to_compare, save_data=True):
    trans_dist, rot_dist = camera_distances(pipeline)
    n_train = trans_dist.shape[0]
    n_test = trans_dist.shape[1]
    per_group = {}
    for g in groups_to_compare:
        tested, tp, ts, rp, rs, tpp, rpp = [], [], [], [], [], [], []
        for j in range(n_test):
            inf = load_influence(results_dir, g, j)
            if inf is None or len(inf) != n_train:
                continue
            t_d, r_d = trans_dist[:, j].numpy(), rot_dist[:, j].numpy()
            a, ap = pearsonr(inf, t_d); b, _ = spearmanr(inf, t_d)
            c, cp = pearsonr(inf, r_d); d, _ = spearmanr(inf, r_d)
            tested.append(j); tp.append(a); ts.append(b); tpp.append(ap)
            rp.append(c); rs.append(d); rpp.append(cp)
        per_group[g] = {"test_indices": np.array(tested),
                        "trans_pearson": np.array(tp), "trans_spearman": np.array(ts),
                        "trans_pearson_p": np.array(tpp), "rot_pearson": np.array(rp),
                        "rot_spearman": np.array(rs), "rot_pearson_p": np.array(rpp)}

    print(f"\n{'='*80}\nPOSE DISTANCE vs INFLUENCE (mean ± std over test images)\n{'-'*80}")
    print(f"{'group':<28}{'trans r':>12}{'trans ρ':>12}{'rot r':>12}{'rot ρ':>12}")
    for g in groups_to_compare:
        d = per_group[g]
        if not len(d["trans_pearson"]):
            continue
        print(f"{g:<28}{d['trans_pearson'].mean():>+12.3f}{d['trans_spearman'].mean():>+12.3f}"
              f"{d['rot_pearson'].mean():>+12.3f}{d['rot_spearman'].mean():>+12.3f}")
    print("=" * 80)
    if save_data:
        torch.save({"per_group": per_group, "trans_dist_matrix": trans_dist,
                    "rot_dist_matrix": rot_dist}, Path(results_dir) / "pose_influence_data.pt")
    return per_group


def plot_pose_influence_single_test(pipeline, results_dir, test_idx, groups_to_compare):
    results_dir = Path(results_dir)
    trans_dist, rot_dist = camera_distances(pipeline)
    t_dist, r_dist = trans_dist[:, test_idx].numpy(), rot_dist[:, test_idx].numpy()
    n_groups = len(groups_to_compare)

    influence_cmap = LinearSegmentedColormap.from_list("influence", ["red", "lightyellow", "green"])

    with plt.rc_context({
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "font.size":        12,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "axes.spines.top":   True,
        "axes.spines.right": True,
    }):
        fig, axes = plt.subplots(2, n_groups, figsize=(5.2 * n_groups, 8.4))
        axes = np.atleast_2d(axes)
        if n_groups == 1:
            axes = axes.reshape(2, 1)

        for col, g in enumerate(groups_to_compare):
            inf = load_influence(results_dir, g, test_idx)
            if inf is None:
                print(f"Missing influence for {g}, test {test_idx}")
                continue

            inf_norm = inf / max(np.abs(inf).max(), 1e-12)   # joint normalisation

            for row, (dist, label) in enumerate(
                [(t_dist, "Translation distance"),
                 (r_dist, "Rotation distance (rad)")]
            ):
                ax = axes[row, col]
                pr, _ = pearsonr(inf, dist)
                sr, _ = spearmanr(inf, dist)

                sc = ax.scatter(dist, inf, s=55, c=inf_norm, cmap=influence_cmap,  vmin=-1, vmax=1, edgecolors="none", zorder=3)

                ax.axhline(0, color="0.6", lw=0.9, ls="--", zorder=1)
                ax.set_xlabel(label)
                ax.set_ylabel(r"Influence $\mathcal{I}(k, j)$")
                ax.set_title(rf"{g}   $r_P$ = {pr:+.2f}   $r_S$ = {sr:+.2f}")
                ax.grid(alpha=0.25, lw=0.6)
                ax.ticklabel_format(style='scientific', axis='y',scilimits=(-2, 2), useMathText=True)

                cb = plt.colorbar(sc, ax=ax)
                cb.set_label(r"Influence (normalised)")

        plt.tight_layout()
        save_path = results_dir / f"pose_influence_test{test_idx}.jpg"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="garden"); a = ap.parse_args()
    S = build_scene(a.scene, build_fim=False)
    compute_pose_influence_correlations(S.pipeline, S.results_dir, S.GROUPS)
    for t in S.TESTS:
        plot_pose_influence_single_test(S.pipeline, S.results_dir, t, S.GROUPS)
