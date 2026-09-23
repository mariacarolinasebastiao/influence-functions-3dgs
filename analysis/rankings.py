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


def compare_rankings_for_test(pipeline, results_dir, test_idx, influence_group="means", top_k=5):
    results_dir = Path(results_dir)
    n_train = len(pipeline.datamanager.train_dataset)
    influence = load_influence(results_dir, influence_group, test_idx)
    if influence is None:
        print(f"Missing influence file for {influence_group}, test {test_idx}")
        return

    data_path = results_dir / "overlap_vs_influence_data.pt"
    overlap = torch.load(data_path)["overlap_matrix"][test_idx].numpy() if data_path.exists() \
        else np.zeros(n_train)
    trans_dist, rot_dist = camera_distances(pipeline)
    t_d, r_d = trans_dist[:, test_idx].numpy(), rot_dist[:, test_idx].numpy()

    metrics = {"influence": influence, "overlap": overlap, "-trans_dist": -t_d, "-rot_dist": -r_d}
    top = {k: set(np.argsort(-v)[:top_k].tolist()) for k, v in metrics.items()}
    bot = {k: set(np.argsort(-v)[-top_k:].tolist()) for k, v in metrics.items()}

    print(f"\n{'='*70}\nTest {test_idx} | group={influence_group} | K={top_k}\n{'='*70}")
    print(f"TOP-{top_k}:")
    for name, s in top.items():
        print(f"  {name:<14}{sorted(s)}")
    print(f"  agreement w/ influence: " + ", ".join(
        f"{n}={len(top['influence'] & top[n])}/{top_k}" for n in ("overlap", "-trans_dist", "-rot_dist")))
    print(f"BOTTOM-{top_k}:")
    for name, s in bot.items():
        print(f"  {name:<14}{sorted(s)}")
    print(f"  agreement w/ influence: " + ", ".join(
        f"{n}={len(bot['influence'] & bot[n])}/{top_k}" for n in ("overlap", "-trans_dist", "-rot_dist")))
    print("Spearman vs influence (all train): " + ", ".join(
        f"{n}={spearmanr(influence, metrics[n])[0]:+.3f}" for n in ("overlap", "-trans_dist", "-rot_dist")))
    print("=" * 70)
    return {"top": top, "bot": bot}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="garden")
    ap.add_argument("--test", type=int, required=True); a = ap.parse_args()
    S = build_scene(a.scene, build_fim=False)
    compare_rankings_for_test(S.pipeline, S.results_dir, a.test, influence_group="means", top_k=5)
