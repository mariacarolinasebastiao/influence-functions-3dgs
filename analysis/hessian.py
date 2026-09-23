import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os
from configs.paths import PROJ_ROOT, WORK_ROOT
sys.path.insert(0, f"{WORK_ROOT}/2026_sebastiao/projects/scripts/splatfacto/fim")
os.chdir(f"{WORK_ROOT}/nerfstudio/nerfstudio")

import os, gc, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from nerfstudio.utils.eval_utils import eval_setup
from src.fim_influence import (FIMInfluenceModule, GSObjective, CustomGSDataset,
                            custom_collate_fn, INFLUENCE_PARAMS,
                            generate_influence_files, influence_vs_loo)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 1e-3
DAMP_FRAC = 0.01
RUN_B = True

config_path = Path(f"{WORK_ROOT}/nerfstudio/nerfstudio/outputs/garden/garden_processed/splatfacto/2026-03-07_184552/config.yml")
os.chdir(f"{PROJ_ROOT}/nerfstudio/nerfstudio")
results_dir = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
GROUPS = ["means", "means + scales", "means + scales + quats", "colors", "5 params (no features_rest)", "all 6 groups"]
TESTS = [3, 9, 13, 14, 16, 17]
LOO_FILES = {j: f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/loo_top10_test{j}/loo_deltas_top10_test{j}.pt" for j in TESTS}

_, pipeline, _, _ = eval_setup(config_path=config_path, test_mode="test")
pipeline.model.to(DEVICE); pipeline.model.eval()
dm = pipeline.datamanager
trainLoader = DataLoader(CustomGSDataset(dm.train_dataset, dm.train_dataset.cameras), batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
testLoader  = DataLoader(CustomGSDataset(dm.eval_dataset,  dm.eval_dataset.cameras),  batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
obj = GSObjective()

module = FIMInfluenceModule(pipeline.model, obj, trainLoader, testLoader, DEVICE, param_names=INFLUENCE_PARAMS, regularization="filter", filter_fraction=ALPHA)
module.build_fim()

# ---------- Question A: how much of the Fisher survives the threshold ----------
print(f"\n{'group':<16}{'kept_count%':>13}{'kept_mass%':>12}{'dropped':>16}{'total':>14}")
for name, raw in module._fim_diag.items():
    cut = ALPHA*raw.mean().clamp(min=1e-30); mask = raw > cut; total = raw.sum().clamp(min=1e-30)
    kc = mask.float().mean().item()*100; km = (raw[mask].sum()/total).item()*100
    print(f"{name:<16}{kc:>13.2f}{km:>12.4f}{(~mask).sum().item():>16,}{raw.numel():>14,}")

names = list(module._fim_diag.keys()); ncol = 3; nrow = int(np.ceil(len(names)/ncol))
with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11}):
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0*ncol, 3.4*nrow)); axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        raw = module._fim_diag[name]; pos = raw[raw > 0]; cut = ALPHA*raw.mean().clamp(min=1e-30); mask = raw > cut
        kc = mask.float().mean().item()*100; km = (raw[mask].sum()/raw.sum().clamp(min=1e-30)).item()*100
        ax.hist(torch.log10(pos.clamp(min=1e-30)).numpy(), bins=80, color="#6E9CC5", edgecolor="none")
        ax.axvline(np.log10(cut.item()), color="#bd310e", lw=1.5, ls="--")
        ax.set_title(rf"{name}  kept {kc:.1f}\% params, {km:.2f}\% mass"); ax.set_xlabel(r"$\log_{10} F_{ii}$"); ax.set_ylabel("count"); ax.grid(alpha=0.25, lw=0.6)
    for ax in axes[len(names):]: ax.set_visible(False)
    plt.tight_layout(); out = results_dir/"fisher_hist.jpg"; plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

# ---------- Question B: do the LOO correlations survive swapping threshold for damping ----------
if RUN_B:
    plt.close("all"); gc.collect(); torch.cuda.empty_cache()
    damp_dir = results_dir.parent/"results_fim_damping"; damp_dir.mkdir(parents=True, exist_ok=True)
    module.regularization = "damping"; module.damping_fraction = DAMP_FRAC
    for name, raw in module._fim_diag.items():
        lam = DAMP_FRAC*raw.mean().clamp(min=1e-30).item()
        module._fim_inv[name] = 1.0/(raw+lam)
    generate_influence_files(module, damp_dir, TESTS, GROUPS, do_dot=False)
    for j in TESTS:
        loo = torch.load(LOO_FILES[j])
        print(f"\n### test {j} ###")
        print("FILTER (alpha=1e-3):"); influence_vs_loo(results_dir, loo, GROUPS, [j])
        print("DAMPING (lambda frac=1e-2):"); influence_vs_loo(damp_dir, loo, GROUPS, [j])
