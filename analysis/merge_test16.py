import torch
from pathlib import Path
from configs.paths import WORK_ROOT
SAVE_DIR = Path(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto/loo_clean_abs")
src = torch.load(f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto/loo_top10_test17/loo_deltas_top10_test17.pt", map_location="cpu")
cf = SAVE_DIR / "loo_deltas_clean_test17.pt"
clean = torch.load(cf, map_location="cpu")
clean[88] = dict(src[88].items())          # use the test-17-folder retrain
torch.save(clean, cf)
print(f"idx 88 test17 -> {clean[88][17]:+.6f}  (now {len(clean)}/10)")
