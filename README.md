# Influence Functions in 3D Gaussian Splatting

Data attribution for 3D Gaussian Splatting: which training images most influence a given
rendered test view, and in which direction, without retraining the model.

**[Project page and interactive scene explorer](https://mcms2.github.io/influence-functions-3dgs/)**

MSc degree project, KTH Royal Institute of Technology.
Supervisors: Mårten Björkman, Marcus Klasson, Vladislav Polianskii.

## What it does

Influence functions estimate the effect of removing a training image on a test view's loss
by combining the two gradients through an inverse Hessian. The Hessian is intractable over
millions of Gaussian parameters, so it is approximated by the diagonal of the empirical
Fisher, which turns each score into an elementwise product:

```
I(z_train, z_test) = (1/N) Σ_i  ∂L(z_test)/∂θ_i · F⁻¹_ii · ∂L(z_train)/∂θ_i
```

Near zero curvature entries are dropped rather than damped, which removes 19% to 45% of the
parameters per group while preserving more than 99.99% of the diagonal Fisher mass. Scores
are computed for six Gaussian attribute groups and validated per test view against leave one
out (LOO) retraining, against camera pose distance and against Gaussian support overlap, on
the Garden, Bicycle and Room scenes of Mip-NeRF 360.

## What came out

Influence agrees with retraining on the predictions it is most confident about, and its
failures are explainable rather than random. On the ten training images of largest absolute
influence, three of six Garden views reach Spearman r_S ≥ 0.73, while uniformly random
samples of training images collapse to no agreement, which is the regime influence itself
calls uninformative. The Fisher term earns its place: on Bicycle 7 a plain gradient dot
product anti-correlates with retraining (−0.19) where the full estimator stays positive
(+0.62). Cheap geometric measures keep up on these curated scenes, but only influence carries
a sign, which is what separates the most beneficial from the most harmful training image when
their geometric overlap is nearly identical. The estimate is also cheap enough to be worth
having: influence over all six parameter groups and all six Garden test views takes about 13
minutes, against a median 56 minutes for a single retrain and roughly 156 hours to attribute
the scene by retraining every one of its 167 training images.

## Layout

| Folder | Purpose |
|---|---|
| `src/` | The method. `fim_influence.py` holds the eFIM influence module and shared helpers. Imported, not run. |
| `configs/` | `paths.py` resolves the filesystem roots; `scenes.py` holds every per scene path and index, plus `build_scene()`. |
| `pipeline/` | Stages that **produce** artifacts from trained models (needs the GPU machine and the checkpoints). |
| `analysis/` | Standalone scripts that **consume** saved `.pt` artifacts to make the thesis figures and numbers. |
| `website/` | The project page, its figures and the three interactive scene explorers. |
| `jobs/` | Shell scripts used to launch training and LOO runs on the GPU machine. |

Checkpoints, `.pt` results and rendered images are not tracked, see `.gitignore`.

## Pipeline order

All stages take `--scene {garden,bicycle,room}`.

1. `pipeline/1_create_loo_frames.py` builds the LOO datasets, symlinking frames and dropping one training image each.
2. `pipeline/2_compute_influence.py` builds the Fisher and writes per group influence scores.
3. `pipeline/3_compute_loo_deltas.py` computes LOO loss deltas from the retrained checkpoints, the ground truth.
4. `pipeline/4_select_loo_indices.py` turns influence scores into the top 5 plus bottom 5 LOO index lists.

Then the `analysis/` scripts read the resulting `.pt` files. Those that need a loaded model
(`pose_influence`, `per_gaussian_influence`, `overlap_baseline`, `rankings`, `loo_psnr`,
`hessian`) also take `--scene`; the rest only read `.pt`.

## Running

Paths are resolved from the environment, with the original layout as the default:

```bash
export GS_WORK_ROOT=/path/to/nerfstudio/projects     # checkpoints and LOO datasets
export GS_PROJ_ROOT=/path/to/results                 # written results, defaults to GS_WORK_ROOT
export GS_NERFSTUDIO_REPO=/path/to/nerfstudio        # the checkout the stages chdir into
```

Then, from the repo root:

```bash
python pipeline/2_compute_influence.py --scene garden
python analysis/overlap_baseline.py --scene garden
```

The per scene indices in `configs/scenes.py` mirror the runs reported in the thesis. The
trained checkpoints and the `.pt` artifacts are not distributed with this repository.

## Cite

```bibtex
@mastersthesis{sebastiao2026influence,
  title   = {Influence Functions in 3D Gaussian Splatting: Data Attribution via the Empirical Fisher},
  author  = {Sebasti{\~a}o, Maria Carolina},
  school  = {KTH Royal Institute of Technology},
  year    = {2026}
}
```
