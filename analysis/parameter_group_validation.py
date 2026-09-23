import torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

from configs.paths import PROJ_ROOT, WORK_ROOT
INF_DIR    = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
GARDEN     = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto")
GARDEN_DSR = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto")
TESTS  = [3, 9, 13, 14, 16, 17]
GROUPS = ["means", "means_scales", "means_scales_quats", "colors", "five_params", "all_groups"]
LABELS = ["means", "m+s", "m+s+q", "colors", "5p", "all-6"]
K2 = 10

# priority order: cleanest source first; earlier-in-list wins on conflict
PRIORITY_GLOBS = [
    "loo_clean_abs/loo_deltas_clean_test*.pt",
    "loo_top10_test*/loo_deltas_abs10_test*.pt",
    "loo_new_retrains/loo_deltas_new15_testall.pt",
    "loo_top10_test*/loo_deltas_top10_test*.pt",
    "loo_deltas_outputs_random.pt",
]

deltas = {}                                  # idx -> {test: delta}; priority-picked
for pat in PRIORITY_GLOBS:
    for root in (GARDEN, GARDEN_DSR):
        for p in sorted(root.glob(pat)):
            try: d = torch.load(p, map_location="cpu")
            except: continue
            for k, vec in d.items():
                if isinstance(vec, dict) and k not in deltas:
                    deltas[k] = vec
print(f"pool: {len(deltas)} unique retrained indices\n")

def load_inf(group, t):
    return torch.load(INF_DIR / group / f"test_image_{t}.pt", map_location="cpu").squeeze().numpy()

def run(mode):                               # mode = "means" | "own"
    res = {}
    for t in TESTS:
        means_inf = load_inf("means", t)
        for g in GROUPS:
            inf_g = load_inf(g, t)
            ref   = means_inf if mode == "means" else inf_g   # what selects the indices
            top10 = np.argsort(np.abs(ref))[::-1][:K2]
            sel   = [int(k) for k in top10 if int(k) in deltas and t in deltas[int(k)]]
            if len(sel) < 6:
                res[(t, g)] = (np.nan, np.nan, len(sel)); continue
            loo = np.array([deltas[k][t] for k in sel])
            x   = np.array([inf_g[k]     for k in sel])        # ALWAYS group g's scores
            res[(t, g)] = (spearmanr(x, loo)[0], pearsonr(x, loo)[0], len(sel))
    return res

def show(res, title, which):                 # which: 0=Spearman, 1=Pearson
    print(title)
    print(f"{'test':>5}  " + "  ".join(f"{l:>12}" for l in LABELS))
    for t in TESTS:
        row = f"{t:>5}  "
        for g in GROUPS:
            rs, rp, n = res[(t, g)]
            v = rs if which == 0 else rp
            cell = f"{v:+.3f}(n{n})" if not np.isnan(v) else f"n={n}"
            row += f"{cell:>12}  "
        print(row)
    print()

res_means = run("means")
res_own   = run("own")

show(res_means, "VERSION 1 — means-selected indices (same set for every group) — SPEARMAN", 0)
show(res_means, "VERSION 1 — means-selected indices (same set for every group) — PEARSON", 1)
show(res_own,   "VERSION 2 — each group's OWN |influence| top-10 — SPEARMAN", 0)
show(res_own,   "VERSION 2 — each group's OWN |influence| top-10 — PEARSON", 1)
