import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
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


def compute_overlap_vs_influence(pipeline, results_dir, objective, group_names):
    """Build (n_test, n_train) gradient-visibility overlap matrix, correlate vs influence per group."""
    results_dir = Path(results_dir)
    model = pipeline.model
    n_train = len(pipeline.datamanager.train_dataset)
    n_test = len(pipeline.datamanager.eval_dataset)

    M_train = []
    for k in tqdm(range(n_train), desc="overlap: train"):
        cam = pipeline.datamanager.train_dataset.cameras[k:k + 1]
        img = pipeline.datamanager.train_dataset[k]["image"]
        M_train.append(contributing_gaussians(model, cam, img, objective))

    overlap = np.full((n_test, n_train), np.nan)
    for j in tqdm(range(n_test), desc="overlap: test"):
        cam = pipeline.datamanager.eval_dataset.cameras[j:j + 1]
        img = pipeline.datamanager.eval_dataset[j]["image"]
        M_j = contributing_gaussians(model, cam, img, objective)
        for k in range(n_train):
            inter = (M_train[k] & M_j).sum().item()
            union = (M_train[k] | M_j).sum().item()
            overlap[j, k] = inter / max(union, 1)

    per_group = {}
    for g in group_names:
        prs, srs, pps, tested = [], [], [], []
        for j in range(n_test):
            inf = load_influence(results_dir, g, j)
            if inf is None or len(inf) != n_train:
                continue
            pr, pp = pearsonr(overlap[j], inf); sr, _ = spearmanr(overlap[j], inf)
            prs.append(pr); srs.append(sr); pps.append(pp); tested.append(j)
        per_group[g] = {"test_indices": tested, "pearson": np.array(prs),
                        "spearman": np.array(srs), "pearson_p": np.array(pps)}

    print(f"\n{'='*70}\nOVERLAP vs INFLUENCE\n{'-'*70}")
    print(f"{'group':<28}{'mean r':>10}{'mean ρ':>10}")
    for g in group_names:
        r, s = per_group[g]["pearson"], per_group[g]["spearman"]
        if len(r):
            print(f"{g:<28}{r.mean():>+10.3f}{s.mean():>+10.3f}")
    print("=" * 70)
    torch.save({"overlap_matrix": torch.tensor(overlap), "per_group": per_group},
               results_dir / "overlap_vs_influence_data.pt")
    return overlap

def overlap_pair(pipeline, train_idx, test_idx, save_path, objective):
    """Visualize the Gaussian overlap of one (train, test) pair on the test image:
    green = Gaussians used by both images, red = used only by the test image."""
    GREEN, RED = 'lime', "#bd310e"   # matching the thesis palette

    model = pipeline.model
    train_cam = pipeline.datamanager.train_dataset.cameras[train_idx:train_idx + 1]
    test_cam = pipeline.datamanager.eval_dataset.cameras[test_idx:test_idx + 1]
    train_img = pipeline.datamanager.train_dataset[train_idx]["image"]
    test_img = pipeline.datamanager.eval_dataset[test_idx]["image"]

    M_k = contributing_gaussians(model, train_cam, train_img, objective).numpy()
    M_j = contributing_gaussians(model, test_cam, test_img, objective).numpy()
    shared, union = int((M_k & M_j).sum()), int((M_k | M_j).sum())
    overlap = shared / max(union, 1)
    print(f"[overlap] train={train_idx} test={test_idx}  shared={shared}  overlap={overlap:.3f}")

    means2d = project_gaussians_for_camera(model, test_cam).cpu().numpy()
    img = test_img.cpu().numpy()
    H, W = img.shape[:2]
    in_frame = (means2d[:, 0] >= 0) & (means2d[:, 0] < W) & \
               (means2d[:, 1] >= 0) & (means2d[:, 1] < H)
    shared_pts = means2d[M_k & M_j & in_frame]
    only_j_pts = means2d[~M_k & M_j & in_frame]

    with plt.rc_context({
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "font.size":        12,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
    }):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img)
        ax.scatter(only_j_pts[:, 0], only_j_pts[:, 1], s=8, c=RED,   alpha=0.45, edgecolors='k', linewidth=0.3,label=r"in test $j$ only")
        ax.scatter(shared_pts[:, 0], shared_pts[:, 1], s=8, c=GREEN, alpha=0.45, edgecolors='k', linewidth=0.3, label=r"shared with train $k$")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=10, markerscale=4,framealpha=0.85)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

def plot_scatter_for_test(results_dir, test_idx, group_names):
    """Scatter of Gaussian overlap vs influence for one test view, one panel per group."""
    results_dir = Path(results_dir)
    data_path = results_dir / "overlap_vs_influence_data.pt"
    if not data_path.exists():
        print(f"Missing {data_path}; run compute_overlap_vs_influence first.")
        return
    overlap = torch.load(data_path)["overlap_matrix"][test_idx].numpy()

    with plt.rc_context({
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "font.size":        12,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "axes.spines.top":   True,
        "axes.spines.right": True,
    }):
        fig, axes = plt.subplots(1, len(group_names), figsize=(5.2 * len(group_names), 4.3))
        axes = np.atleast_1d(axes)

        for ax, g in zip(axes, group_names):
            inf = load_influence(results_dir, g, test_idx)
            if inf is None:
                ax.set_visible(False); continue
            pr, _ = pearsonr(overlap, inf)
            sr, _ = spearmanr(overlap, inf)

            ax.scatter(overlap, inf, s=55, alpha=0.7, color="#6E9CC5", edgecolors="none", zorder=3)

            ax.axhline(0, color="0.6", lw=0.9, ls="--", zorder=1)
            ax.set_xlabel(r"Gaussian overlap $(k, j)$")
            ax.set_ylabel(r"Influence $\mathcal{I}(k, j)$")
            ax.set_title(rf"{g}   $r_P$ = {pr:+.2f}   $r_S$ = {sr:+.2f}")
            ax.grid(alpha=0.25, lw=0.6)
            ax.ticklabel_format(style='scientific', axis='y',scilimits=(-2, 2), useMathText=True)

        plt.tight_layout()
        save_path = results_dir / f"overlap_vs_influence_test{test_idx}.jpg"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="garden"); a = ap.parse_args()
    S = build_scene(a.scene, build_fim=False)
    GROUPS = ["means", "means + scales", "colors"]
    compute_overlap_vs_influence(S.pipeline, S.results_dir, S.obj, GROUPS)
    for t in S.TESTS:
        plot_scatter_for_test(S.results_dir, t, GROUPS)
