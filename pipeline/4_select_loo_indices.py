import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import numpy as np
from pathlib import Path

from configs.scenes import SCENES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden", choices=["garden", "bicycle", "room"])
    ap.add_argument("--test", type=int, required=True)
    ap.add_argument("--group", default="means")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()

    influence_dir = Path(SCENES[a.scene]["results_dir"]) / a.group
    K = a.top_k

    scores = torch.load(influence_dir / f"test_image_{a.test}.pt")
    if scores.ndim == 2:
        scores = scores[:, 0]
    scores = scores.cpu().numpy()

    # top-K beneficial + top-K harmful (signed)
    top_beneficial = np.argsort(scores)[::-1][:K].tolist()
    top_harmful = np.argsort(scores)[:K].tolist()
    signed = top_beneficial + top_harmful

    # top-2K by |influence|
    abs2k = np.argsort(np.abs(scores))[::-1][:K * 2].tolist()

    print(f"Test {a.test} | group: {a.group}\n")
    print(f"VERSION 1 — top-{K} beneficial + top-{K} harmful (signed):")
    print("  beneficial: ", [(i, f"{scores[i]:+.4e}") for i in top_beneficial])
    print("  harmful:    ", [(i, f"{scores[i]:+.4e}") for i in top_harmful])
    print(f"  LOO_TRAIN_INDICES = {signed}\n")

    print(f"VERSION 2 — top-{K * 2} by |influence|:")
    for idx in abs2k:
        print(f"  train_idx={idx:3d}  influence={scores[idx]:+.4e}")
    print(f"  LOO_TRAIN_INDICES = {abs2k}\n")

    print(f"Overlap between versions: {set(signed) & set(abs2k)}")


if __name__ == "__main__":
    main()
