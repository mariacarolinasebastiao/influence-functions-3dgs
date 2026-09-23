import torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from configs.paths import PROJ_ROOT, WORK_ROOT
INF_DIR = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
LOO_DIR = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto")
TESTS  = [3, 9, 13, 14, 16, 17]
GROUPS = ["means", "means_scales", "means_scales_quats", "colors", "five_params", "all_groups"]
K = 5    # how many extremes to characterize the tails


def load_inf(g, j):
    p = INF_DIR / g / f"test_image_{j}.pt"
    return torch.load(p).squeeze().numpy() if p.exists() else None


def load_loo(j):
    p = LOO_DIR / f"loo_top10_test{j}/loo_deltas_top10_test{j}.pt"
    return torch.load(p) if p.exists() else None


print(f"\n{'group':<22}{'test':>5}{'top+ mean':>12}{'top- mean':>12}{'ratio':>8}{'rho_full':>10}{'rho_top':>9}{'rho_bot':>9}")
print("-" * 95)
for j in TESTS:
    loo = load_loo(j)
    for g in GROUPS:
        inf = load_inf(g, j)
        if inf is None:
            print(f"{g:<22}{j:>5}  (missing influence)")
            continue
        order = np.argsort(inf)
        top_k_pos = inf[order[-K:]]
        top_k_neg = inf[order[:K]]
        ratio = float(np.mean(np.abs(top_k_neg)) / max(np.mean(top_k_pos), 1e-12))

        rho_full = rho_top = rho_bot = float("nan")
        if loo is not None:
            valid = sorted(k for k in loo if j in loo[k])
            if len(valid) >= 3:
                x = inf[valid]; y = np.array([loo[k][j] for k in valid])
                rho_full = spearmanr(x, y)[0]
                # split valid into positive-influence and negative-influence halves
                pos_v = [k for k in valid if inf[k] > 0]
                neg_v = [k for k in valid if inf[k] <= 0]
                if len(pos_v) >= 3:
                    rho_top = spearmanr(inf[pos_v], np.array([loo[k][j] for k in pos_v]))[0]
                if len(neg_v) >= 3:
                    rho_bot = spearmanr(inf[neg_v], np.array([loo[k][j] for k in neg_v]))[0]

        print(f"{g:<22}{j:>5}{top_k_pos.mean():>+12.2e}{top_k_neg.mean():>+12.2e}{ratio:>8.2f}"
              f"{rho_full:>+10.3f}{rho_top:>+9.3f}{rho_bot:>+9.3f}")
    print("-" * 95)
