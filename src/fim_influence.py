import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from nerfstudio.utils.eval_utils import eval_setup
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from gsplat.cuda._wrapper import fully_fused_projection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


INFLUENCE_PARAMS = ["means", "opacities", "scales", "features_dc", "features_rest", "quats"]

ABLATION_GROUPS = {
    "means":                       ["means"],
    "means + scales":              ["means", "scales"],
    "means + scales + quats":      ["means", "scales", "quats"],
    "colors":                      ["features_dc", "features_rest"],
    "features_dc only":            ["features_dc"],
    "5 params (no features_rest)": ["means", "scales", "quats", "opacities", "features_dc"],
    "all 6 groups":                ["means", "scales", "quats", "features_dc", "opacities", "features_rest"],
}
GROUP_FOLDERS = {
    "means":                       "means",
    "means + scales":              "means_scales",
    "means + scales + quats":      "means_scales_quats",
    "colors":                      "colors",
    "features_dc only":            "features_dc",
    "5 params (no features_rest)": "five_params",
    "all 6 groups":                "all_groups",
}


class CustomGSDataset(Dataset):
    def __init__(self, data, cameras):
        self.data = data
        self.cameras = cameras

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        camera = self.cameras[idx: idx + 1]
        return {"camera": camera, "image": item["image"]}


def custom_collate_fn(batch):
    return batch[0]


class FIMInfluenceModule:
    def __init__(self, model, objective, train_loader, test_loader, device,
                 param_names=None, damping_fraction=0.01,
                 regularization="damping", filter_fraction=0.0):
        self.model = model
        self.objective = objective
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.damping_fraction = damping_fraction
        self.regularization = regularization
        self.filter_fraction = filter_fraction
        self.param_names = param_names if param_names is not None else INFLUENCE_PARAMS

        self.tracked_params = {n: p for n, p in model.named_parameters()
            if any(n == pn or n.endswith("." + pn) for pn in self.param_names) and p.requires_grad}
        print(f"[FIM] Tracking: {list(self.tracked_params.keys())}")
        print(f"[FIM] Total params: {sum(p.numel() for p in self.tracked_params.values()):,}")

        self._train_gradients = None
        self._fim_diag = None
        self._fim_inv = None
        self._n_train = None

    def _transfer_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._transfer_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._transfer_to_device(v) for v in batch]
        elif hasattr(batch, "to"):
            return batch.to(self.device)
        return batch

    def _compute_grad(self, batch):
        self.model.zero_grad()
        batch = self._transfer_to_device(batch)
        outputs = self.objective.train_outputs(self.model, batch)
        loss = self.objective.train_loss_on_outputs(outputs, batch)
        loss.backward()
        grads = {}
        for name, param in self.tracked_params.items():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu()
            else:
                grads[name] = torch.zeros(param.shape, dtype=param.dtype)
        return grads

    def build_fim(self):
        if self._fim_diag is not None:
            return

        print("[FIM] Building FIM diagonal (one pass over training data)...")
        self._train_gradients = []
        accumulator = defaultdict(lambda: None)
        for batch in tqdm(self.train_loader, desc="FIM: train grads"):
            grads = self._compute_grad(batch)
            self._train_gradients.append(grads)
            for name, g in grads.items():
                accumulator[name] = g ** 2 if accumulator[name] is None else accumulator[name] + g ** 2

        self._n_train = len(self._train_gradients)
        n = self._n_train
        print(f"[FIM] Regularization = {self.regularization}")
        self._fim_diag, self._fim_inv = {}, {}

        for name, acc in accumulator.items():
            raw = acc / n
            self._fim_diag[name] = raw

            if self.regularization == "damping":
                lam = self.damping_fraction * raw.mean().clamp(min=1e-30).item()
                self._fim_inv[name] = 1.0 / (raw + lam)
                print(f"{name}: mean={raw.mean():.3e} max={raw.max():.3e} damping={lam:.3e}")

            elif self.regularization == "filter":
                cutoff = self.filter_fraction * raw.mean().clamp(min=1e-30).item()
                mask = raw > cutoff
                inv = torch.zeros_like(raw)
                inv[mask] = 1.0 / raw[mask]
                self._fim_inv[name] = inv
                print(f"{name}: mean={raw.mean():.3e} max={raw.max():.3e} cutoff={cutoff:.3e} "
                      f"kept={mask.float().mean()*100:.1f}% (dropped {(~mask).sum().item():,}/{raw.numel():,})")
            else:
                raise ValueError(f"Unknown regularization: {self.regularization}")
        print("[FIM] Done.")

    def influences(self, train_indices, test_indices, use_fim=True):
        """Influence scores [len(train), len(test)].  I = (1/N) g_test · F^-1 · g_train."""
        self.build_fim()
        n = self._n_train
        scores = torch.zeros(len(train_indices), len(test_indices))
        test_batches = list(self.test_loader)
        for j, test_idx in enumerate(tqdm(test_indices, desc="Computing influence")):
            test_grad = self._compute_grad(test_batches[test_idx])
            for i, train_idx in enumerate(train_indices):
                score = 0.0
                for name in self.tracked_params:
                    g_test = test_grad[name]
                    g_train = self._train_gradients[train_idx][name]
                    if use_fim:
                        score += (g_test * self._fim_inv[name] * g_train).sum().item()
                    else:
                        score += (g_test * g_train).sum().item()
                scores[i, j] = score / n
        return scores


class GSObjective:
    def train_outputs(self, model, batch):
        return model(batch["camera"])["rgb"]

    def train_loss_on_outputs(self, outputs, batch):
        gt = batch["image"].to(outputs.device)
        l1 = torch.abs(gt - outputs).mean()
        ssim_loss = 1 - ssim(gt.permute(2, 0, 1)[None], outputs.permute(2, 0, 1)[None],
                             data_range=1.0, size_average=True)
        return 0.8 * l1 + 0.2 * ssim_loss

    def train_regularization(self, params):
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)

    def test_loss(self, model, params, batch):
        return self.train_loss_on_outputs(self.train_outputs(model, batch), batch)


def camera_distances(pipeline):
    """Return (trans_dist, rot_dist), each shape (n_train, n_test)."""
    train_c2w = pipeline.datamanager.train_dataset.cameras.camera_to_worlds.cpu()
    eval_c2w  = pipeline.datamanager.eval_dataset.cameras.camera_to_worlds.cpu()
    n_train, n_test = train_c2w.shape[0], eval_c2w.shape[0]
    trans = torch.cdist(train_c2w[:, :3, 3], eval_c2w[:, :3, 3], p=2)
    traces = train_c2w[:, :3, :3].reshape(n_train, 9) @ eval_c2w[:, :3, :3].reshape(n_test, 9).T
    rot = torch.acos(((traces - 1) / 2).clamp(-1, 1))
    return trans, rot


def load_influence(results_dir, group, test_idx, prefix=""):
    """Load a saved per-group influence vector, or None if missing. Always 1-D."""
    path = Path(results_dir) / GROUP_FOLDERS.get(group, group) / f"{prefix}test_image_{test_idx}.pt"
    if not path.exists():
        return None
    return torch.load(path).squeeze().numpy()


def compute_influences_for_groups(module, group_subset, test_idx, train_indices, use_fim=True):
    """Influence over a subset of parameter groups, using the module's regularized inverse."""
    names = [n for n in module.tracked_params
             if any(n == g or n.endswith("." + g) for g in group_subset)]
    if not names:
        print(f"  WARNING: no matching params for {group_subset}")
        return None
    n = module._n_train
    test_grad = module._compute_grad(list(module.test_loader)[test_idx])
    scores = []
    for train_idx in train_indices:
        s = 0.0
        for name in names:
            g_test, g_train = test_grad[name], module._train_gradients[train_idx][name]
            s += (g_test * module._fim_inv[name] * g_train).sum().item() if use_fim \
                 else (g_test * g_train).sum().item()
        scores.append(s / n)
    return np.array(scores)


def generate_influence_files(module, results_dir, test_indices, groups, do_dot=True):
    """For each group and test image, compute influence over ALL train images and save to
    results_dir/<group_folder>/test_image_{j}.pt  (+ dot_test_image_{j}.pt if do_dot)."""
    results_dir = Path(results_dir)
    module.build_fim()
    train_indices = list(range(module._n_train))
    for group in groups:
        folder = results_dir / GROUP_FOLDERS[group]
        folder.mkdir(parents=True, exist_ok=True)
        for j in test_indices:
            fim = compute_influences_for_groups(module, ABLATION_GROUPS[group], j, train_indices, use_fim=True)
            torch.save(torch.tensor(fim, dtype=torch.float32), folder / f"test_image_{j}.pt")
            if do_dot:
                dot = compute_influences_for_groups(module, ABLATION_GROUPS[group], j, train_indices, use_fim=False)
                torch.save(torch.tensor(dot, dtype=torch.float32), folder / f"dot_test_image_{j}.pt")
        print(f"[gen] {group}: saved {len(test_indices)} test image(s) -> {folder}")


def _valid_train(loo_deltas, test_idx):
    return [k for k in sorted(loo_deltas) if test_idx in loo_deltas[k]]


@torch.no_grad()
def project_gaussians_for_camera(model, camera):
    """(x, y) pixel coords of every Gaussian center as seen by this camera (for plotting)."""
    camera = camera.to(DEVICE)
    R = camera.camera_to_worlds[0, :3, :3]
    T = camera.camera_to_worlds[0, :3, 3:4]
    R_edit = torch.diag(torch.tensor([1., -1., -1.], device=DEVICE)) @ R.T
    T_edit = -R_edit @ T
    viewmat = torch.eye(4, device=DEVICE)
    viewmat[:3, :3] = R_edit
    viewmat[:3, 3:4] = T_edit
    viewmat = viewmat[None]
    K = camera.get_intrinsics_matrices().to(DEVICE)
    W, H = int(camera.width.item()), int(camera.height.item())
    quats = model.quats / model.quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(model.scales)
    _, means2d, _, _, _ = fully_fused_projection(model.means, None, quats, scales, viewmat, K, W, H,
                                                 eps2d=0.3, packed=False, near_plane=0.01, far_plane=1e10)
    return means2d[0]


def contributing_gaussians(model, camera, image, objective):
    """A Gaussian contributes if any of its params gets a non-zero gradient under this image."""
    device = model.means.device
    camera, image = camera.to(device), image.to(device)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_(); p.grad.zero_()
    loss = objective.train_loss_on_outputs(objective.train_outputs(model, {"camera": camera, "image": image}), {"camera": camera, "image": image})
    loss.backward()
    N = model.means.shape[0]
    M = torch.zeros(N, dtype=torch.bool, device=device)
    for name in ["means", "scales", "quats", "opacities", "features_dc", "features_rest"]:
        param = getattr(model, name, None)
        if param is None or param.grad is None:
            continue
        M |= (param.grad.abs().view(N, -1).sum(dim=-1) > 0)
    return M.cpu()


