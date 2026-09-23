import os, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13})
from configs.paths import PROJ_ROOT, WORK_ROOT
os.chdir(f"{PROJ_ROOT}/nerfstudio/nerfstudio")

SCENES = {
    "garden":  (f"{WORK_ROOT}/nerfstudio/nerfstudio/outputs/garden/garden_processed/splatfacto/2026-03-07_184552/config.yml", [3, 9, 13, 14, 16, 17]),
    "bicycle": (f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/bicycle_processed/splatfacto/2026-05-11_160007/config.yml", [7]),
    "room":    (f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/baseline/room_processed/splatfacto/2026-06-05_153133/config.yml", [28]),
}
OUT_DIR = Path(f"{WORK_ROOT}/2026_sebastiao/projects/residual_maps")
CMAP, MODE = "inferno", "heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_config(spec):
    p = Path(spec)
    if p.name == "config.yml": return p
    bad = ("loo", "remove", "random", "warm", "noise")
    cands = [c for c in p.glob("**/config.yml") if list((c.parent / "nerfstudio_models").glob("step-*.ckpt")) and not any(b in str(c).lower() for b in bad)]
    if not cands: raise FileNotFoundError(f"no baseline config under {p} — set it explicitly")
    proc = [c for c in cands if "processed" in str(c).lower()]
    return sorted(proc or cands)[-1]

def render_view(pl, dm, j):
    cam = dm.eval_dataset.cameras[j:j+1].to(pl.device)
    gt = dm.eval_dataset[j]["image"].to(pl.device)
    with torch.no_grad(): pred = pl.model.get_outputs_for_camera(cam)["rgb"]
    return gt.clamp(0, 1).cpu().numpy(), pred.clamp(0, 1).cpu().numpy()

def save_figure(scene, j, gt, pred, vmax):
    res = np.abs(gt - pred).mean(-1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(gt); ax[0].set_title("Ground truth")
    ax[1].imshow(pred); ax[1].set_title("Prediction")
    if MODE == "overlay":
        ax[2].imshow(gt.mean(-1), cmap="gray")
        im = ax[2].imshow(res, cmap=CMAP, vmin=0, vmax=vmax, alpha=np.clip(res / vmax, 0, 1))
    else:
        im = ax[2].imshow(res, cmap=CMAP, vmin=0, vmax=vmax)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    cax = ax[2].inset_axes([0.0, 1.04, 1.0, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cb.set_label("|GT - pred|")
    out = OUT_DIR / f"{scene}_test{j}_residual.jpg"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", out)

for scene, (spec, views) in SCENES.items():
    cfg = find_config(spec)
    print(f"\n=== {scene}: {cfg} ===")
    _, pl, _, _ = eval_setup(cfg, test_mode="test", update_config_callback=lambda c: setattr(c, "load_step", None) or c)
    pl.eval(); dm = pl.datamanager
    data = {}
    for j in views:
        if j >= len(dm.eval_dataset): print(f"  skip test {j}: scene has only {len(dm.eval_dataset)} eval views"); continue
        data[j] = render_view(pl, dm, j)
    if data:
        vmax = np.percentile(np.concatenate([np.abs(g - p).mean(-1).ravel() for g, p in data.values()]), 99)
        for j, (g, p) in data.items(): save_figure(scene, j, g, p, vmax)
    del pl; torch.cuda.empty_cache()
