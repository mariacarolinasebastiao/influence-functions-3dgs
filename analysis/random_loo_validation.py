import torch, numpy as np
from pathlib import Path

# ============ EDIT ============
from configs.paths import PROJ_ROOT
RES   = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames")
CACHE = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/pooled_random_deltas.pt")
TESTS = [3, 9, 13, 14, 16, 17]
GROUP = "means"

random_idx = sorted(torch.load(CACHE)["indices"])
n_rand = len(random_idx)

print(f"For each test view, count of the {n_rand} random indices falling in the\n"
      f"extreme bands of the SIGNED influence distribution.\n")
print(f"{'test':>5}{'N':>6}"
      f"{'top-3% rand':>14}{'bot-3% rand':>14}{'either-3% rand':>17}"
      f"{'top-10% rand':>15}{'bot-10% rand':>15}{'either-10% rand':>18}")
print("-"*104)
for j in TESTS:
    inf = torch.load(RES / GROUP / f"test_image_{j}.pt").squeeze().numpy()
    N = len(inf)
    order = np.argsort(inf)                       # ascending signed
    rank = np.empty(N, dtype=int)
    rank[order] = np.arange(N)
    pct = 100.0 * (rank + 1) / N                   # signed-influence percentile

    rand_pct = pct[random_idx]
    top3  = int(np.sum(rand_pct >= 97))            # top 3% (matches top-5 of 167)
    bot3  = int(np.sum(rand_pct <= 3))             # bottom 3%
    top10 = int(np.sum(rand_pct >= 90))            # top 10%
    bot10 = int(np.sum(rand_pct <= 10))            # bottom 10%

    print(f"{j:>5}{N:>6}"
          f"{top3:>9}/{n_rand}{bot3:>9}/{n_rand}{top3+bot3:>12}/{n_rand}"
          f"{top10:>10}/{n_rand}{bot10:>10}/{n_rand}{top10+bot10:>13}/{n_rand}")
