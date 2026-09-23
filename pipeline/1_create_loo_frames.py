import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import json, shutil, argparse
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup

from configs.scenes import SCENES, NERFSTUDIO_REPO


def make(train_idx, train_names, eval_names, base, ORIG, LOO_BASE, scene):
    remove = train_names[train_idx]
    out = LOO_BASE / f"{scene}_data_remove{train_idx}"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "images_4").mkdir()
    data = json.loads(json.dumps(base))
    tr, ev = [], []
    for fr in data["frames"]:
        name = Path(fr["file_path"]).name
        s, s4 = ORIG / "images" / name, ORIG / "images_4" / name
        if name in eval_names:
            new = f"eval_{name}"
            fr["file_path"] = f"./images/{new}"
            os.symlink(s, out / "images" / new); os.symlink(s4, out / "images_4" / new); ev.append(fr)
        elif name == remove:
            continue
        else:
            new = f"train_{name}"
            fr["file_path"] = f"./images/{new}"
            os.symlink(s, out / "images" / new); os.symlink(s4, out / "images_4" / new); tr.append(fr)
    data["frames"] = tr + ev
    json.dump(data, open(out / "transforms.json", "w"), indent=4)
    print(f"  remove {train_idx} ({remove}): {len(tr)} train + {len(ev)} eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden", choices=["garden", "bicycle", "room"])
    a = ap.parse_args()
    cfg = SCENES[a.scene]

    os.chdir(NERFSTUDIO_REPO)
    CONFIG = Path(cfg["config_path"])
    LOO_BASE = Path(cfg["loo_base"])
    TRAIN_INDICES = cfg["loo_frame_indices"]

    _, pipeline, _, _ = eval_setup(CONFIG, test_mode="test")
    dm = pipeline.datamanager

    ORIG = Path(dm.train_dataset.image_filenames[0]).resolve().parent.parent
    assert (ORIG / "transforms.json").exists(), f"no transforms.json in {ORIG}"
    assert (ORIG / "images").is_dir() and (ORIG / "images_4").is_dir(), f"missing images/ or images_4/ in {ORIG}"
    print("ORIG =", ORIG)

    train_names = [Path(f).name for f in dm.train_dataset.image_filenames]
    eval_names = set(Path(f).name for f in dm.eval_dataset.image_filenames)
    del pipeline

    base = json.load(open(ORIG / "transforms.json"))
    for k in TRAIN_INDICES:
        make(k, train_names, eval_names, base, ORIG, LOO_BASE, a.scene)
    print("done ->", LOO_BASE)


if __name__ == "__main__":
    main()
