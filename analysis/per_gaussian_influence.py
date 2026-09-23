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


def per_gaussian_influence(module, group_subset, test_idx, train_idx, signed=True):
    """Decompose ONE (train -> test) influence score across the scene's Gaussians.

    Returns a (G,) array. Entry g = how much of train image `train_idx`'s influence
    on test view `test_idx` flows through Gaussian g.
        signed=True  -> +/- : through this Gaussian the image HELPS (+) or HURTS (-).
                        Sums back to the scalar influence (it's the same number, split up).
        signed=False -> |.| : involvement magnitude. Does NOT sum back to the scalar.
    Same formula as compute_influences_for_groups; the ONLY change is we sum each
    Gaussian's params separately (view(G, -1).sum(-1)) instead of summing everything.
    """
    names = [n for n in module.tracked_params
             if any(n == g or n.endswith("." + g) for g in group_subset)]
    if not names:
        print(f"  WARNING: no matching params for {group_subset}")
        return None
    G = module.model.means.shape[0]
    n = module._n_train
    test_grad = module._compute_grad(list(module.test_loader)[test_idx])
    per_g = torch.zeros(G)
    for name in names:
        g_test  = test_grad[name]
        g_train = module._train_gradients[train_idx][name]
        term = (g_test * module._fim_inv[name] * g_train).view(G, -1)   # (G, params_of_this_name)
        per_g += term.sum(dim=-1) if signed else term.abs().sum(dim=-1)
    return (per_g / n).numpy()


def plot_per_gaussian_influence(pipeline, module, train_idx, test_idx, save_path, group_subset=("means",), signed=True, clip_pct=99):
    """Overlay the per-Gaussian influence of (train_idx -> test_idx) on the test image.
    Signed: green = this Gaussian makes the train image HELP the test view, red = HURT.
    The unsigned magnitude version (signed=False) is the involvement heat map.
    Mirrors overlap_pair, but colors by influence instead of shared/not-shared."""
    model = pipeline.model
    test_cam = pipeline.datamanager.eval_dataset.cameras[test_idx:test_idx + 1]
    test_img = pipeline.datamanager.eval_dataset[test_idx]["image"]

    infl = per_gaussian_influence(module, list(group_subset), test_idx, train_idx, signed=signed)
    print(f"[per-gauss] train={train_idx} test={test_idx}  " f"sum over Gaussians = {infl.sum():+.3e}   (should match scalar influence if signed)")

    means2d = project_gaussians_for_camera(model, test_cam).cpu().numpy()
    img = test_img.cpu().numpy()
    H, W = img.shape[:2]
    in_frame = (means2d[:, 0] >= 0) & (means2d[:, 0] < W) & \
        (means2d[:, 1] >= 0) & (means2d[:, 1] < H)
    pts, val = means2d[in_frame], infl[in_frame]

    # draw the strongest Gaussians on top so they aren't buried
    order = np.argsort(np.abs(val))
    pts, val = pts[order], val[order]

    if signed:
        lim  = max(np.percentile(np.abs(val), clip_pct), 1e-30)
        cmap = LinearSegmentedColormap.from_list("influence", ["#bd310e", "lightyellow", "lime"])
        norm = Normalize(vmin=-lim, vmax=lim)
        cbar_label = r"per-Gaussian influence (signed)"
    else:
        lim  = max(np.percentile(val, clip_pct), 1e-30)
        cmap, norm = plt.cm.viridis, Normalize(vmin=0, vmax=lim)
        cbar_label = r"per-Gaussian influence (magnitude)"

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12}):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img)
        sc = ax.scatter(pts[:, 0], pts[:, 1], s=8, c=val, cmap=cmap, norm=norm,
                        alpha=0.6, edgecolors="k", linewidth=0.2, zorder=3)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"Train $k={train_idx}\rightarrow$ test $j={test_idx}$   "
                     rf"({'+'.join(group_subset)})")
        cb = plt.colorbar(sc, ax=ax); cb.set_label(cbar_label)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="garden")
    ap.add_argument("--train", type=int, required=True); ap.add_argument("--test", type=int, required=True)
    a = ap.parse_args()
    S = build_scene(a.scene, build_fim=True)
    plot_per_gaussian_influence(S.pipeline, S.module, train_idx=a.train, test_idx=a.test,
        save_path=S.results_dir / f"pergauss_train{a.train}_test{a.test}.jpg",
        group_subset=("means",), signed=True)
