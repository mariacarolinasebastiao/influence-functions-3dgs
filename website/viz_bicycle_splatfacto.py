import numpy as np
import torch
from pathlib import Path
import plotly.graph_objects as go
import os
import base64
import json
import io
from PIL import Image
from tqdm import tqdm

from nerfstudio.utils.eval_utils import eval_setup

from configs.paths import WORK_ROOT
CONFIG_PATH = Path(f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/bicycle_processed/splatfacto/2026-05-11_160007/config.yml")
RESULTS_DIR = Path(f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/results_fim_all_frames")
OUTPUT_HTML = Path(f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/influence_viz_bicycle.html")
IMG_MAX_WIDTH = 480  # resize images before embedding — larger = better quality but bigger file

os.chdir(f"{WORK_ROOT}/nerfstudio/nerfstudio")

_, pipeline, _, _ = eval_setup(config_path=CONFIG_PATH, test_mode="test")
datamanager = pipeline.datamanager

train_c2w = datamanager.train_dataset.cameras.camera_to_worlds.cpu().numpy()
test_c2w  = datamanager.eval_dataset.cameras.camera_to_worlds.cpu().numpy()
train_pos = train_c2w[:, :3, 3]
test_pos  = test_c2w[:, :3, 3]
N_train   = len(train_pos)
N_test    = len(test_pos)
print(f"Train: {N_train} | Test: {N_test}")

# ── parameter groups: auto-discover every subfolder holding test_image_*.pt ────
DISPLAY_NAMES = {
    "means": "means", "scales": "scales", "quats": "quats",
    "opacities": "opacities", "colors": "colors",
    "features_dc": "features_dc", "features_rest": "features_rest",
    "means_scales": "means + scales", "means_scales_quats": "means + scales + quats",
    "5p": "5p", "all6": "all-6", "all_6": "all-6",
}
def discover_groups(root):
    found = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(sub.glob("test_image_*.pt")):
            found[DISPLAY_NAMES.get(sub.name, sub.name)] = sub.name
    if not found and any(root.glob("test_image_*.pt")):
        found["means"] = ""
    return found
GROUPS = discover_groups(RESULTS_DIR)
print(f"Available groups: {list(GROUPS)}")

# ── normalization (consistent with scatter plots: divide by global abs-max) ───
def normalize(scores):
    denom = max(np.abs(scores).max(), 1e-12)
    return (scores / denom).tolist()

# ── load influence scores for every group ─────────────────────────────────────
def load_group(folder):
    """Return (raw [N_test][N_train], colors [N_test][N_train])."""
    raw, cols = [], []
    for i in range(N_test):
        pt = RESULTS_DIR / folder / f"test_image_{i}.pt"
        if pt.exists():
            arr = torch.load(pt).squeeze().numpy()
        else:
            arr = np.zeros(N_train)
        raw.append(arr.tolist())
        cols.append(normalize(arr))
    return raw, cols

all_raw    = {}   # group -> [[N_train] raw scores per test]
all_colors = {}   # group -> [[N_train] normalized per test]

for name, folder in GROUPS.items():
    r, c = load_group(folder)
    all_raw[name]    = r
    all_colors[name] = c
    print(f"  Loaded '{name}' from {folder or './'}/")

default_group = next(iter(all_colors))

# ── encode images as base64 ───────────────────────────────────────────────────
def encode_image(path, max_width=IMG_MAX_WIDTH):
    img = Image.open(str(path)).convert("RGB")
    w, h = img.size
    if w > max_width:
        img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

print("Encoding train images...")
train_b64 = [encode_image(p) for p in tqdm(datamanager.train_dataset.image_filenames)]
print("Encoding test images...")
test_b64  = [encode_image(p) for p in tqdm(datamanager.eval_dataset.image_filenames)]

# ── plotly figure ─────────────────────────────────────────────────────────────
COLORSCALE = [[0.0, "#2166ac"], [0.5, "#d9d9d9"], [1.0, "#b2182b"]]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=train_pos[:, 0], y=train_pos[:, 1], z=train_pos[:, 2],
    mode="markers",
    marker=dict(
        size=4,
        color=all_colors[default_group][0],
        colorscale=COLORSCALE,
        cmin=-1, cmax=1,
        colorbar=dict(title="Influence<br>(normalized)", thickness=14, x=1.0),
        showscale=True,
    ),
    customdata=list(range(N_train)),
    name="Train cameras",
    hovertemplate="Train %{customdata}<extra></extra>",
))

fig.add_trace(go.Scatter3d(
    x=test_pos[:, 0], y=test_pos[:, 1], z=test_pos[:, 2],
    mode="markers+text",
    marker=dict(size=7, color="black", symbol="diamond"),
    text=[str(i) for i in range(N_test)],
    textposition="top center",
    name="Test cameras",
    hovertemplate="Test %{text}<extra></extra>",
))

fig.update_layout(
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
    margin=dict(l=0, r=0, b=0, t=10),
    legend=dict(x=0.01, y=0.99),
    autosize=True,
)

plot_div = fig.to_html(full_html=False, include_plotlyjs=True, div_id="plotly-plot",
                       config={"responsive": True})

# ── HTML template ─────────────────────────────────────────────────────────────
html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FIM Influence Visualization — Bicycle</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: sans-serif; background: #1c1c1c; color: #eee; }
  #topbar {
    display: flex; align-items: center; gap: 16px;
    padding: 8px 16px; background: #2a2a2a; border-bottom: 1px solid #444;
    height: 44px;
  }
  #topbar label { font-size: 14px; }
  select {
    padding: 4px 10px; font-size: 14px; border-radius: 4px;
    background: #444; color: #eee; border: 1px solid #666; cursor: pointer;
  }
  #hover-info { font-size: 12px; color: #aaa; margin-left: auto; }
  #main {
    display: grid;
    grid-template-columns: 1fr 360px;
    height: calc(100vh - 44px);
  }
  #plot-container { overflow: hidden; }
  #plot-container > div { width: 100% !important; height: 100% !important; }
  #side-panel {
    display: flex; flex-direction: column;
    gap: 10px; padding: 10px; background: #222;
    border-left: 1px solid #444; overflow-y: auto;
  }
  .card { background: #2a2a2a; border-radius: 6px; padding: 10px; border: 1px solid #3a3a3a; }
  .card h4 { font-size: 12px; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .card img { width: 100%; border-radius: 4px; display: block; }
  .card .empty { font-size: 12px; color: #555; text-align: center; padding: 30px 0; }
  .score { font-size: 13px; font-weight: bold; margin-top: 6px; }
  .score.pos { color: #d6604d; }
  .score.neg { color: #6baed6; }
</style>
</head>
<body>
<div id="topbar">
  <label>Test Image: <select id="test-select"></select></label>
  <label>Group: <select id="group-select"></select></label>
  <span id="hover-info">Hover over a training camera to see its image</span>
</div>
<div id="main">
  <div id="plot-container">PLOT_DIV_PLACEHOLDER</div>
  <div id="side-panel">
    <div class="card">
      <h4>Test Image <span id="test-label">—</span></h4>
      <img id="test-img" src="" />
    </div>
    <div class="card" id="train-card">
      <h4>Training Image <span id="train-label">—</span></h4>
      <div id="train-empty" class="empty">Hover over a training camera</div>
      <img id="train-img" src="" style="display:none;" />
      <div id="train-score" class="score"></div>
    </div>
  </div>
</div>
<script>
const trainImages = TRAIN_IMAGES_JSON;
const testImages  = TEST_IMAGES_JSON;
const allColors   = ALL_COLORS_JSON;   // { groupName: [[N_train normed] per test] }
const rawScores   = RAW_SCORES_JSON;   // { groupName: [[N_train raw]   per test] }
const groupNames  = GROUP_NAMES_JSON;  // [groupName, ...]

const testSelect  = document.getElementById('test-select');
const groupSelect = document.getElementById('group-select');

for (let i = 0; i < testImages.length; i++) {
  const opt = document.createElement('option');
  opt.value = i; opt.text = 'Test ' + i;
  testSelect.appendChild(opt);
}
for (const g of groupNames) {
  const opt = document.createElement('option');
  opt.value = g; opt.text = g;
  groupSelect.appendChild(opt);
}

let currentTest  = 0;
let currentGroup = groupNames[0];

function updatePlot() {
  document.getElementById('test-img').src = testImages[currentTest];
  document.getElementById('test-label').textContent = currentTest;
  Plotly.restyle('plotly-plot', { 'marker.color': [allColors[currentGroup][currentTest]] }, [0]);
}

testSelect.addEventListener('change',  () => { currentTest  = parseInt(testSelect.value); updatePlot(); });
groupSelect.addEventListener('change', () => { currentGroup = groupSelect.value;          updatePlot(); });
updatePlot();

const plotDiv = document.getElementById('plotly-plot');

plotDiv.on('plotly_hover', function(data) {
  const pt = data.points[0];
  if (pt.curveNumber !== 0) return;
  const tidx  = pt.customdata;
  const score = rawScores[currentGroup][currentTest][tidx];

  document.getElementById('train-img').src             = trainImages[tidx];
  document.getElementById('train-img').style.display   = 'block';
  document.getElementById('train-empty').style.display = 'none';
  document.getElementById('train-label').textContent   = tidx;

  const scoreEl = document.getElementById('train-score');
  scoreEl.textContent = 'score: ' + score.toExponential(3);
  scoreEl.className   = 'score ' + (score >= 0 ? 'pos' : 'neg');

  document.getElementById('hover-info').textContent =
    '[' + currentGroup + ']  Train ' + tidx + '  →  influence: ' + score.toExponential(3);
});

plotDiv.on('plotly_unhover', function() {
  document.getElementById('hover-info').textContent = 'Hover over a training camera to see its image';
});
</script>
</body>
</html>"""

html = html.replace("PLOT_DIV_PLACEHOLDER", plot_div)
html = html.replace("TRAIN_IMAGES_JSON",    json.dumps(train_b64))
html = html.replace("TEST_IMAGES_JSON",     json.dumps(test_b64))
html = html.replace("ALL_COLORS_JSON",      json.dumps(all_colors))
html = html.replace("RAW_SCORES_JSON",      json.dumps(all_raw))
html = html.replace("GROUP_NAMES_JSON",     json.dumps(list(all_colors.keys())))

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"Saved: {OUTPUT_HTML}")
print(f"File size: {OUTPUT_HTML.stat().st_size / 1e6:.1f} MB")
