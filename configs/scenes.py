import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from nerfstudio.utils.eval_utils import eval_setup

from src.fim_influence import (
    FIMInfluenceModule, GSObjective, CustomGSDataset, custom_collate_fn,
    INFLUENCE_PARAMS, DEVICE,
)

from configs.paths import WORK_ROOT, PROJ_ROOT, NERFSTUDIO_REPO

DEFAULT_GROUPS = [
    "means", "means + scales", "means + scales + quats",
    "colors", "5 params (no features_rest)", "all 6 groups",
]

REG = dict(regularization="filter", filter_fraction=0.001)

# Verify these paths against the machine before running; they mirror the originals.
SCENES = {
    "garden": {
        "config_path": f"{WORK_ROOT}/nerfstudio/nerfstudio/outputs/garden/garden_processed/splatfacto/2026-03-07_184552/config.yml",
        "proj_dir":    f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto",
        "results_dir": f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/results_fim_all_frames",
        "TESTS": [3, 9, 13, 14, 16, 17],
        "LOO_FILES": {
            t: f"{PROJ_ROOT}/2026_sebastiao/projects/garden_splatfacto/loo_top10_test{t}/loo_deltas_top10_test{t}.pt"
            for t in [3, 9, 13, 14, 16, 17]
        },
        "LOO_SETS": {
            3:  [68, 67, 6, 36, 37, 69, 120, 95, 121, 70],
            9:  [34, 33, 92, 146, 119, 91, 93, 32, 145, 3],
            13: [47, 130, 156, 48, 103, 17, 129, 102, 131, 104],
            14: [111, 139, 57, 112, 140, 138, 84, 85, 163, 164],
            16: [131, 130, 156, 50, 49, 157, 48, 18, 106, 105],
            17: [165, 166, 88, 143, 164, 141, 140, 113, 90, 63],
        },
        "loo_base": f"{WORK_ROOT}/2026_sebastiao/projects/garden_splatfacto/loo_new_retrains",
        "loo_frame_indices": [69, 70, 95, 120, 121, 3, 32, 93, 145, 17, 102, 104, 129, 90, 141],
    },
    "bicycle": {
        "config_path": f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/bicycle_processed/splatfacto/2026-05-11_160007/config.yml",
        "proj_dir":    f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto",
        "results_dir": f"{PROJ_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/results_fim_zerofilter",
        "TESTS": [3, 7],
        "LOO_FILES": {
            t: f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/loo_top10_test{t}/loo_deltas_top10_test{t}.pt"
            for t in [3, 7]
        },
        "LOO_SETS": {7: [28, 70, 73, 24, 113]},
        "loo_base": f"{WORK_ROOT}/2026_sebastiao/projects/bicycle_splatfacto/loo_test7_data",
        "loo_frame_indices": [28, 70, 73, 24, 113],
    },
    "room": {
        "config_path": f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/baseline/room_processed/splatfacto/2026-06-05_153133/config.yml",
        "proj_dir":    f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto",
        "results_dir": f"{PROJ_ROOT}/2026_sebastiao/projects/room_splatfacto/results_fim",
        "TESTS": [28],
        "LOO_FILES": {28: f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_clean_abs/loo_deltas_clean_test28.pt"},
        "LOO_SETS": {28: [261, 262, 14, 17, 15, 263, 126, 265, 124, 127]},
        "loo_base": f"{WORK_ROOT}/2026_sebastiao/projects/room_splatfacto/loo_test28_data",
        "loo_frame_indices": [263, 126, 265, 124, 127],
    },
}


def build_scene(scene, build_fim=True):
    cfg = SCENES[scene]
    os.chdir(NERFSTUDIO_REPO)
    _, pipeline, _, _ = eval_setup(config_path=Path(cfg["config_path"]), test_mode="test")
    pipeline.model.to(DEVICE)
    pipeline.model.eval()
    dm = pipeline.datamanager

    train_loader = DataLoader(CustomGSDataset(dm.train_dataset, dm.train_dataset.cameras),
                              batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(CustomGSDataset(dm.eval_dataset, dm.eval_dataset.cameras),
                             batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    obj = GSObjective()
    module = FIMInfluenceModule(pipeline.model, obj, train_loader, test_loader, DEVICE,
                                param_names=INFLUENCE_PARAMS, **REG)
    if build_fim:
        module.build_fim()

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        scene=scene, pipeline=pipeline, dm=dm, obj=obj, module=module,
        results_dir=results_dir, config_path=Path(cfg["config_path"]),
        proj_dir=Path(cfg["proj_dir"]), TESTS=cfg["TESTS"],
        LOO_FILES=cfg.get("LOO_FILES", {}), LOO_SETS=cfg.get("LOO_SETS", {}),
        GROUPS=cfg.get("GROUPS", DEFAULT_GROUPS),
    )
