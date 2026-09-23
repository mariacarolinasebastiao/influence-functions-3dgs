import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
from pytorch_msssim import ssim

from configs.scenes import SCENES, NERFSTUDIO_REPO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_test_losses(config_path):
    _, pipeline, _, _ = eval_setup(config_path, test_mode="test", update_config_callback=lambda config: setattr(config, 'load_step', None) or config)
    pipeline.eval()
    device = pipeline.device
    dm = pipeline.datamanager
    losses = []
    for test_idx in range(len(dm.eval_dataset)):
        camera = dm.eval_dataset.cameras[test_idx:test_idx + 1].to(device)
        gt = dm.eval_dataset[test_idx]["image"].to(device)
        with torch.no_grad():
            pred = pipeline.model.get_outputs_for_camera(camera)["rgb"]
            l1 = torch.abs(gt - pred).mean()
            ssim_loss = 1.0 - ssim(gt.permute(2, 0, 1)[None], pred.permute(2, 0, 1)[None], data_range=1.0, size_average=True)
            losses.append((0.8 * l1 + 0.2 * ssim_loss).item())
    return losses


def find_loo_config(train_idx, proj, scene):
    cands = [c for c in proj.glob(f"**/{scene}_*remove{train_idx}/**/config.yml")
             if list((c.parent / "nerfstudio_models").glob("*.ckpt"))]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room", choices=["garden", "bicycle", "room"])
    a = ap.parse_args()
    cfg = SCENES[a.scene]

    os.chdir(NERFSTUDIO_REPO)
    proj = Path(cfg["proj_dir"])
    loo_sets = cfg["LOO_SETS"]
    save_dir = proj / "loo_deltas_clean"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Computing baseline losses (once)...")
    baseline_losses = compute_test_losses(Path(cfg["config_path"]))

    unique_idx = sorted({i for s in loo_sets.values() for i in s})
    full_deltas = {}
    for train_idx in unique_idx:
        cfg_loo = find_loo_config(train_idx, proj, a.scene)
        if cfg_loo is None:
            print(f"  MISSING model for idx {train_idx}"); continue
        try:
            loo_losses = compute_test_losses(cfg_loo)
            full_deltas[train_idx] = {t: loo_losses[t] - baseline_losses[t] for t in range(len(baseline_losses))}
            print(f"  idx {train_idx:3d} done")
        except Exception as e:
            print(f"  Error idx {train_idx}: {e}")

    for t, idxs in loo_sets.items():
        clean = {i: full_deltas[i] for i in idxs if i in full_deltas}
        out = save_dir / f"loo_deltas_clean_test{t}.pt"
        torch.save(clean, out)
        miss = [i for i in idxs if i not in full_deltas]
        print(f"  test {t}: saved {len(clean)}/{len(idxs)} -> {out}"
              + (f"   MISSING {miss}" if miss else ""))


if __name__ == "__main__":
    main()
