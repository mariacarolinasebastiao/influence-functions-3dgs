import torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# influence score dirs
from configs.paths import PROJ_ROOT, WORK_ROOT
GARDEN_INF = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
BICY_INF   = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/results_fim_zerofilter")
ROOM_INF   = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/room_splatfacto/results_fim_all_frames")

# ONE clean |influence| delta file per test
GARDEN_CLEAN = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto/loo_clean_abs")
BICY_CLEAN   = Path(f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/loo_clean_abs")
ROOM_CLEAN   = Path(f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_clean_abs")

SCENES = [
    dict(name="Garden",  inf=GARDEN_INF, test=14, clean=GARDEN_CLEAN/"loo_deltas_clean_test14.pt"),
    dict(name="Garden",  inf=GARDEN_INF, test=17, clean=GARDEN_CLEAN/"loo_deltas_clean_test17.pt"),
    dict(name="Garden",  inf=GARDEN_INF, test=3,  clean=GARDEN_CLEAN/"loo_deltas_clean_test3.pt"),
    dict(name="Garden",  inf=GARDEN_INF, test=9,  clean=GARDEN_CLEAN/"loo_deltas_clean_test9.pt"),
    dict(name="Garden",  inf=GARDEN_INF, test=13, clean=GARDEN_CLEAN/"loo_deltas_clean_test13.pt"),
    dict(name="Garden",  inf=GARDEN_INF, test=16, clean=GARDEN_CLEAN/"loo_deltas_clean_test16.pt"),
    dict(name="Bicycle", inf=BICY_INF,   test=7,  clean=BICY_CLEAN/"loo_deltas_clean_test7.pt"),
    dict(name="Room",    inf=ROOM_INF,   test=28, clean=ROOM_CLEAN/"loo_deltas_clean_test28.pt"),
]

GROUPS = ["means", "means_scales", "means_scales_quats", "colors", "five_params", "all_groups"]

def load_scores(inf_dir, group, test, prefix=""):
    p = inf_dir / group / f"{prefix}test_image_{test}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu")
    if isinstance(d, torch.Tensor):
        return d.float().squeeze().numpy()
    keys = sorted(d.keys())
    return np.array([float(d[k]) for k in keys])

print(f"{'scene/test':<14}{'group':<20}{'n':>3}  "
      f"{'eFIM rS':>8} {'dot rS':>8}   {'eFIM rP':>8} {'dot rP':>8}")
print("-" * 80)

for s in SCENES:
    name, inf_dir, test = s["name"], s["inf"], s["test"]
    clean = torch.load(s["clean"], map_location="cpu")     # {idx: {test: delta}}
    idxs    = sorted(clean.keys())                          # the |influence| top-10
    loo_arr = np.array([clean[k][test] for k in idxs])
    n = len(idxs)

    for g in GROUPS:
        efim = load_scores(inf_dir, g, test, prefix="")
        dot  = load_scores(inf_dir, g, test, prefix="dot_")
        if efim is None or dot is None:
            continue
        ef_s = spearmanr([efim[k] for k in idxs], loo_arr)[0]
        dt_s = spearmanr([dot[k]  for k in idxs], loo_arr)[0]
        ef_p = pearsonr([efim[k] for k in idxs], loo_arr)[0]
        dt_p = pearsonr([dot[k]  for k in idxs], loo_arr)[0]
        print(f"{name+' '+str(test):<14}{g:<20}{n:>3}  "
              f"{ef_s:>+8.3f} {dt_s:>+8.3f}   {ef_p:>+8.3f} {dt_p:>+8.3f}")
    print()
