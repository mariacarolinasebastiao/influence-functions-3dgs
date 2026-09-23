import os, torch, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from nerfstudio.utils.eval_utils import eval_setup
from pytorch_msssim import ssim

from configs.paths import PROJ_ROOT, WORK_ROOT
os.chdir(f"{PROJ_ROOT}/nerfstudio/nerfstudio")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ EDIT ============
BASELINE_CFG = f"{WORK_ROOT}/nerfstudio/nerfstudio/outputs/garden/garden_processed/splatfacto/2026-03-07_184552/config.yml"
RANDOM_ROOT  = Path(f"{WORK_ROOT}/nerfstudio/nerfstudio/outputs/")   # has garden_loo_random_remove_train{k}
INF_DIR      = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
CACHE        = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/pooled_random_deltas.pt")
NOISE_FLOOR  = {3: 0.00038, 9: 0.00007, 13: 0.00011, 14: 0.00087, 17: 0.00015}   # 1σ_seed from your noise floor; test 16 unknown
TESTS  = [3, 9, 13, 14, 16, 17]
GROUPS = ["means", "means_scales", "means_scales_quats", "colors", "five_params", "all_groups"]   # folder names

def find_cfg(d):
    cands = [c for c in Path(d).glob("**/config.yml") if (c.parent/"nerfstudio_models").exists()]
    assert cands, f"no completed run in {d}"
    return max(cands, key=lambda p: p.stat().st_mtime)

def test_losses(cfg):
    _, pl, _, _ = eval_setup(Path(cfg), test_mode="test",
        update_config_callback=lambda c: setattr(c, "load_step", None) or c)
    pl.eval(); dm = pl.datamanager; out = []
    for j in range(len(dm.eval_dataset)):
        cam = dm.eval_dataset.cameras[j:j+1].to(pl.device)
        gt  = dm.eval_dataset[j]["image"].to(pl.device)
        with torch.no_grad():
            pred = pl.model.get_outputs_for_camera(cam)["rgb"]
            l1 = torch.abs(gt-pred).mean()
            ss = 1.0 - ssim(gt.permute(2,0,1)[None], pred.permute(2,0,1)[None], data_range=1.0, size_average=True)
            out.append((0.8*l1 + 0.2*ss).item())
    del pl; torch.cuda.empty_cache()
    return np.array(out)

if not CACHE.exists():
    print("computing baseline losses…")
    base = test_losses(find_cfg(Path(BASELINE_CFG).parent))
    indices, deltas = [], []
    for d in sorted(RANDOM_ROOT.glob("garden_loo_random_remove_train*")):
        k = int(d.name.split("train")[-1])
        try: loo = test_losses(find_cfg(d))
        except Exception as e: print(f"  skip {k}: {e}"); continue
        indices.append(k); deltas.append(loo - base)
        print(f"  idx {k:>3}: delta[14]={loo[14]-base[14]:+.6f}")
    torch.save({"indices": indices, "deltas": np.stack(deltas), "baseline": base}, CACHE)
    print(f"cached -> {CACHE}")

D = torch.load(CACHE)
indices = D["indices"]; deltas = D["deltas"]   # (n_runs, n_test)
print(f"\nLoaded {len(indices)} random LOO runs: {indices}")

# ablation
print(f"\n{'group':<22}{'test':>5}{'r_P':>9}{'rho':>9}{'|d|range×10³':>16}{'#|d|>2σ':>10}")
print("-"*72)
for j in TESTS:
    d_j = deltas[:, j]
    sigma = NOISE_FLOOR.get(j, None)
    rng = d_j.max() - d_j.min()
    n_sig = int(np.sum(np.abs(d_j) > 2*sigma)) if sigma else -1
    for g in GROUPS:
        inf_path = INF_DIR / g / f"test_image_{j}.pt"
        if not inf_path.exists():
            print(f"{g:<22}{j:>5}  (no influence file)"); continue
        inf_all = torch.load(inf_path).squeeze().numpy()
        inf_j = inf_all[indices]
        rp = pearsonr(inf_j, d_j)[0]
        rs = spearmanr(inf_j, d_j)[0]
        print(f"{g:<22}{j:>5}{rp:>+9.3f}{rs:>+9.3f}{rng*1e3:>16.3f}{n_sig:>10}")
    print("-"*72)
