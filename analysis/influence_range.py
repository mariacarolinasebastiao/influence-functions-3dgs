import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from configs.paths import PROJ_ROOT
INF_DIR = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/room_splatfacto/results_fim_all_frames")
GROUP   = "all_groups"
SCENE   = "Room"
N_VIEWS = 31

GREEN, RED = "#0ca120", "#bd310e"

with plt.rc_context({"font.family":      "serif","mathtext.fontset": "cm","font.size":        12,
    "axes.labelsize":   12, "axes.titlesize":   13,}):
    fig, ax = plt.subplots(figsize=(9, 7))

    for j in range(N_VIEWS):
        inf_file = INF_DIR / GROUP / f"test_image_{j}.pt"
        if not inf_file.exists():
            continue
        inf = torch.load(inf_file).squeeze().numpy()
        max_val = inf.max()
        min_val = inf.min()

        ax.hlines(j, min_val, max_val, color="0.6", lw=1.0, zorder=1)
        ax.scatter(max_val, j, color=GREEN, s=70, zorder=3,edgecolors="k", linewidth=0.3)
        ax.scatter(min_val, j, color=RED,   s=70, zorder=3,edgecolors="k", linewidth=0.3)

    ax.axvline(0, color="0.4", lw=0.9, linestyle="--", zorder=2)
    ax.set_yticks(range(N_VIEWS))
    ax.set_yticklabels([f"Test {j}" for j in range(N_VIEWS)])
    ax.set_xlabel(r"Influence score (all)")
    ax.invert_yaxis()                          # test 0 at top, last at bottom
    ax.grid(axis="x", alpha=0.3, lw=0.6)

    plt.tight_layout()
    plt.savefig(f"influence_range_{SCENE.lower()}_{GROUP}.jpg", dpi=200, bbox_inches="tight")
    plt.close()

print("Saved")
