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


@torch.no_grad()
def _render_psnr(pipeline, test_idx):
    pipeline.model.to(DEVICE); pipeline.model.eval()
    cam = pipeline.datamanager.eval_dataset.cameras[test_idx:test_idx + 1].to(DEVICE)
    gt = pipeline.datamanager.eval_dataset[test_idx]["image"].to(DEVICE)
    pred = pipeline.model.get_outputs_for_camera(cam)["rgb"]
    mse = ((pred - gt) ** 2).mean().clamp(min=1e-10)
    return (-10.0 * torch.log10(mse)).item()

def compute_loo_psnr(test_idx, full_config_path, loo_config_for, train_indices, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        c = torch.load(cache_path)
        if c["test_idx"] == test_idx:
            return c["psnr_full"], c["psnrs"]
    _, pf, _, _ = eval_setup(config_path=full_config_path, test_mode="test")
    psnr_full = _render_psnr(pf, test_idx)
    del pf; torch.cuda.empty_cache()
    psnrs = {}
    for k in tqdm(list(train_indices), desc=f"LOO PSNR (test {test_idx})"):
        cfg = loo_config_for(k)
        if cfg is None or not Path(cfg).exists():
            print(f"  missing LOO model for {k}"); continue
        _, pl, _, _ = eval_setup(config_path=cfg, test_mode="test")
        psnrs[k] = _render_psnr(pl, test_idx)
        del pl; torch.cuda.empty_cache()
    if cache_path is not None:
        torch.save({"test_idx": test_idx, "psnr_full": psnr_full, "psnrs": psnrs}, cache_path)
    return psnr_full, psnrs


def plot_loo_psnr_per_train(psnr_full, psnrs, test_idx, save_path, influence=None):
    from matplotlib.lines import Line2D
    ks = sorted(psnrs.keys())
    vals = np.array([psnrs[k] for k in ks])
    if influence is not None:
        colors = ['#0ca120' if influence[k] > 0 else '#bd310e' for k in ks]
    else:
        colors = ['steelblue' if v < psnr_full else 'firebrick' for v in vals]
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm","font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,"axes.spines.top": True, "axes.spines.right": True}):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(ks, vals, s=130, c=colors, marker='*', edgecolors='k', linewidth=0.4, zorder=3)
        ax.axhline(psnr_full, color='black', lw=1, alpha=0.7)
        ax.set_xlabel(r"Removed training image $k$")
        ax.set_ylabel("PSNR on test (dB)")
        ax.grid(True, alpha=0.3)
        handles = [Line2D([0],[0], marker='*', color='w', markerfacecolor='#0ca120', markeredgecolor='k', markersize=13, label="influence beneficial"),
            Line2D([0],[0], marker='*', color='w', markerfacecolor='#bd310e', markeredgecolor='k', markersize=13, label="influence harmful"),
            Line2D([0],[0], color='black', lw=1, alpha=0.7, label=f"baseline = {psnr_full:.2f} dB"),]
        ax.legend(handles=handles, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="garden")
    ap.add_argument("--test", type=int, required=True); a = ap.parse_args()
    S = build_scene(a.scene, build_fim=False)
    def loo_config_for(k):
        cands = [c for c in S.proj_dir.glob(f"**/*loo*remove{k}/**/config.yml")
                 if list((c.parent / "nerfstudio_models").glob("*.ckpt"))]
        return max(cands, key=lambda p: p.stat().st_mtime) if cands else None
    train_indices = S.LOO_SETS.get(a.test, [])
    psnr_full, psnrs = compute_loo_psnr(a.test, S.config_path, loo_config_for, train_indices)
    plot_loo_psnr_per_train(psnr_full, psnrs, a.test, S.results_dir / f"loo_psnr_test{a.test}.jpg")
