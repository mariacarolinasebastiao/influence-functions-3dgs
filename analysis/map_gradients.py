#!/usr/bin/env python3

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import time
from tqdm import tqdm
from nerfstudio.utils.eval_utils import eval_setup


def save_outputs(output_root: Path, stem: str, rgb: torch.Tensor, grad_map: torch.Tensor, args):
    # save original RGB
    rgb_np = rgb.detach().cpu().numpy()
    plt.imsave(output_root / "rgb" / f"{stem}.png", rgb_np)

    # saving the gradients as .npy (
    raw = grad_map.detach().cpu().numpy()
    np.save(output_root / "grads" / f"{stem}.npy", raw)

    # doing the log scaling, 
    if args.save_mpl:
        mpl_img = raw.astype(np.float32, copy=True)
        if args.log_scale:
            mpl_img = np.log1p(mpl_img)

        if args.pclip and args.pclip > 0:
            vmax = np.percentile(mpl_img, args.pclip)
            vmin = np.percentile(mpl_img, 0)
            mpl_img = np.clip(mpl_img, vmin, vmax)

        plt.imsave(output_root / "grads" / f"{stem}_mpl.png", mpl_img, cmap="magma")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="outputs/unnamed/splatfacto/main/config.yml")
    parser.add_argument("--out", type=str, default="outputs/gradient_maps_simple")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--save-mpl", action="store_true", help="Save matplotlib visualization in grads/")
    parser.add_argument("--log-scale", action="store_true", help="Use log1p scaling for matplotlib output")
    parser.add_argument("--pclip", type=float, default=99.5, help="Percentile clip for matplotlib output (0 to disable)")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    output_root = Path(args.out)

    # make output dirs
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rgb").mkdir(parents=True, exist_ok=True)
    (output_root / "grads").mkdir(parents=True, exist_ok=True)

    repo_root = config_path.parents[0] if config_path.is_absolute() else Path.cwd()
    os.chdir(str(repo_root))

    torch.set_grad_enabled(True) # enable gradients
    _, pipeline, _, _ = eval_setup(config_path=config_path, test_mode="val", ) 
    datamanager = pipeline.datamanager
    pipeline.to("cuda") 
    pipeline.model.eval()
    pipeline.model.requires_grad_(True)

    parameters = tuple(pipeline.model.parameters())     # i need the parameters of the model


    # get training cameras and filenames through datamanager and dataparser
    cameras = datamanager.train_dataparser_outputs.cameras.to("cuda")
    filenames = datamanager.train_dataparser_outputs.image_filenames

    if args.max_images <= 0: #all
        number_images = len(filenames)
    else:#all or the number specified
        number_images = min(len(filenames), args.max_images)  


    # loop over (all) training images
    for i in tqdm(range(number_images), desc="Gradient maps"):
        image_time_start = time.perf_counter()
        cam = cameras[i : i + 1]

        # C_i is the pixel's value computed when rendering
        outputs_of_the_model = pipeline.model.get_outputs(cam.to(pipeline.device))
        rgb = outputs_of_the_model["rgb"].view(cam.image_height, cam.image_width, 3).clamp(0.0, 1.0)

        h, w, _ = rgb.shape
        pixels_flat = rgb.reshape(-1, 3)
        grad_map = torch.zeros(pixels_flat.shape[0], device=rgb.device, dtype=rgb.dtype) #empty

        # i want to sum all partial derivatives over all pixels in all training images
        total_pixels = pixels_flat.shape[0]
    

        for idx in range(total_pixels):
            scalar = pixels_flat[idx].sum()  # im summing all my rgb channels /C_i to get a scalar
            grads = torch.autograd.grad(scalar,parameters, retain_graph=i != total_pixels - 1, allow_unused=True,)#the derivatives of C_i wrt all model parameters
            pixel_grad_l1 = torch.tensor(0.0, device=rgb.device, dtype=rgb.dtype) #initialize a variable to store the sum of all gradients for this pixel
            for g in grads:
                if g is None:
                    continue
                pixel_grad_l1 += g.abs().sum()  # sum absolute gradients for this pixel
            grad_map[idx] = pixel_grad_l1.detach()

        # i need to save that sum and put it in a map, same height and width as the image
        grad_map = grad_map.reshape(h, w)

        stem = Path(filenames[i]).stem or f"{i:04d}"
        save_outputs(output_root, stem, rgb, grad_map, args)

        elapsed = time.perf_counter() - image_time_start
        print(f"Image {i + 1}/{number_images} done in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
