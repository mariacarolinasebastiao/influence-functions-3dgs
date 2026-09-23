import torch, numpy as np, re
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup

from configs.paths import PROJ_ROOT, WORK_ROOT
ROOM_INF = Path(f"{PROJ_ROOT}/2026_sebastiao/projects/room_splatfacto/results_fim_all_frames")
BASELINE_CONFIG = Path(f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/baseline/room_processed/splatfacto/2026-06-05_153133/")
GROUP, PREFIX, TOP_N = "means", "", 10

def load(p):
    d = torch.load(p, map_location="cpu")
    if isinstance(d, torch.Tensor):
        return np.arange(len(d.float().squeeze())), d.float().squeeze().numpy()
    keys = sorted(d.keys()); return np.array(keys), np.array([float(d[k]) for k in keys])

files = sorted((ROOM_INF / GROUP).glob(f"{PREFIX}test_image_*.pt"))
mat = {}
for f in files:
    keys, vals = load(f)
    for k, v in zip(keys, vals): mat.setdefault(int(k), []).append(float(v))

rows = []
for k, vs in mat.items():
    vs = np.array(vs); rows.append((k, vs.mean(), (vs < 0).mean(), vs.min(), len(vs)))
rows.sort(key=lambda r: r[1])

_, pipeline, _, _ = eval_setup(BASELINE_CONFIG / "config.yml", test_mode="inference")
train_files = pipeline.datamanager.train_dataset.image_filenames
name = lambda k: train_files[k].name if 0 <= k < len(train_files) else f"<oor {k}>"

print(f"Room  group={GROUP}  prefix='{PREFIX}'  n_test_views={len(files)}")
print(f"  {'k':>4}  {'mean':>9}  {'frac_neg':>8}  {'min':>9}  {'nviews':>6}  file")
for k, m, fn, mn, nv in rows[:TOP_N]:
    print(f"  {k:>4}  {m:>+9.4f}  {fn:>8.2f}  {mn:>+9.4f}  {nv:>6}  {name(k)}")
