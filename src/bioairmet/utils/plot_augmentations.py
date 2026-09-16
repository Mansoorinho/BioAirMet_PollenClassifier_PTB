'''
@file    :   plot_augmentations.py
@author  :   Mansoor Nabawi
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   Visualize the effect of EACH image augmentation, separately, on
    real holographic images.

    For every input image this script renders one figure:
      * row 0            : the original image (repeated as a reference)
      * one row per transform (horizontal_flip ... cutout), each row showing
        ``--draws`` random applications of that transform ALONE (enabled with
        prob=1.0, every other transform disabled, so you see its isolated
        effect)
      * final row        : the FULL custom-mode pipeline built from the
        ``image_transforms`` block of a stage config (its real per-transform
        probabilities), ``--draws`` times.

    All rows are produced by the actual training code path
    (``ImageAugmentation_Generic`` in custom mode), so what you see is what
    training applies. No training runs and no HDF5 data is loaded; the script
    only reads the image files you pass it.

    Usage (from the repository root):
        python src/bioairmet/utils/plot_augmentations.py \\
            --images /path/to/a.rec_mag.png /path/to/b.rec_mag.png \\
            --draws 3 --seed 0 --domain tensor
    or, with the package installed:
        python -m bioairmet.utils.plot_augmentations --images ... --draws 3

    Flags:
      --images    holographic PNG files to visualize (default: two sample
                  rec_mag.png files; pass your own for your corpus)
      --out       output directory for the PNG figures
                  (default: ./plots/augmentation_effects relative to the CWD)
      --draws     random draws per transform (default: 3)
      --seed      RNG seed (default: 0)
      --domain    'tensor' = float [0,1] (the 'minmax'/'global' readers) or
                  'pil' = 8-bit grayscale (the 'legacy' reader).
                  Use the domain of the reader your training runs with.
      --config    stage config whose image_transforms block feeds the
                  'full pipeline' row (default: the SSL template)

    How to use this to tune augmentation values:
      1. Run the script on a couple of representative images of your corpus,
         in the domain of the reader you train with.
      2. Inspect each row: a good transform is clearly visible, keeps the
         particle centered and recognizable, and keeps the white background
         white.
      3. Adjust the matching entry in the config's image_transforms block
         (prob / std / sigma / gamma / size / angle / ...) and re-run to
         compare; repeat until every row looks like a plausible, augmented
         member of the corpus.  The 'full pipeline' row must look like a
         normal augmented version of the original.
      4. Remember the effective probability of a transform is
         img_aug_prob x prob (the reader gate is applied first).

    Outputs: <out>/<image-stem>.png
'''

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt

from bioairmet.data.augmentation import ImageAugmentation_Generic




# The script lives in src/bioairmet/utils/, so the package root (src/bioairmet)
# is one level up; this keeps the default config path working both from the
# source tree and from an installed package (the config YAMLs are package data).
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "ssl" / "SSL_config_general.yaml"
# Output lands in <CWD>/plots/augmentation_effects; run from the repository
# root to write into the repository's plots/ directory.
DEFAULT_OUT = Path("plots") / "augmentation_effects"
DEFAULT_IMAGES = [
    "/home/test0/9017437f_2026-06-02_10h/"
    "poleno-34_2026-06-02_10.39.56.906599_ev.computed_data.holography.image_pairs.0.1.rec_mag.png",
    "/home/test0/9017437f_2026-06-02_10h/"
    "poleno-34_2026-06-02_10.39.56.906599_ev.computed_data.holography.image_pairs.0.0.rec_mag.png",
]

# The transforms shown, one row each, in pipeline order.  Parameters are the
# representative values used for the visualization (prob is forced to 1.0 so
# every row actually shows the effect).  Keep these in sync with the stage
# config you are tuning.
TRANSFORM_PARAMS = {
    "horizontal_flip":  {},
    "vertical_flip":    {},
    "rotation":         {"angle": 180, "translation": [0.05, 0.05],
                         "interpolation": "nearest", "translate_round": True},
    "affine":           {"translation": [0.1, 0.1], "scale": [0.9, 1.1]},
    "color_jitter":     {"brightness": 0.5, "contrast": 0.5},
    "gaussian_blur":    {"kernel_size": 3, "sigma": [0.1, 2.0]},
    "gaussian_noise":   {"std": 0.05},
    "speckle_noise":    {"sigma": 0.05},
    "gamma_correction": {"gamma": [0.7, 1.3]},
    "cutout":           {"size": [0.02, 0.06], "clear_center": 0.6},
}


def load_image(path: str, domain: str):
    """Load a 16-bit holographic PNG and normalize it exactly like the readers.

    domain='tensor' -> (1, H, W) float32 tensor in [0, 1] (the 'minmax' reader)
    domain='pil'    -> 8-bit grayscale PIL image (the 'legacy' reader)
    """
    with Image.open(path) as pil_img:
        arr = np.array(pil_img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi - lo > 1e-5 else np.full_like(arr, 0.5)
    if domain == "tensor":
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def to_display(img) -> np.ndarray:
    """Any accepted image type -> (H, W) float array in [0, 1] for imshow."""
    if isinstance(img, Image.Image):
        return np.array(img, dtype=np.float32) / 255.0
    return img[0].numpy()


def build_single_transform(name: str, seed: int) -> ImageAugmentation_Generic:
    """Custom-mode pipeline with ONLY ``name`` enabled (prob 1.0)."""
    cfg = {
        "parallel_flipping_rotation": True,
        "fill": "border",  # same as the shipped stage template default
        name: {"enabled": True, "prob": 1.0, **TRANSFORM_PARAMS[name]},
    }
    return ImageAugmentation_Generic(cfg, legacy=False, stage="ssl",
                                     gate_prob=None, seed=seed)


def build_full_pipeline(config_path: Path, seed: int) -> ImageAugmentation_Generic:
    """Custom-mode pipeline from the image_transforms block of a stage config."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    block = cfg["data"]["augmentation"]["image_transforms"]
    return ImageAugmentation_Generic(block, legacy=False, stage="ssl",
                                     gate_prob=None, seed=seed)


def plot_image(image_path: str, domain: str, draws: int, seed: int,
               config_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    original = load_image(image_path, domain)

    rows = ["Original"]
    grid = [to_display(original)] * draws
    for name in TRANSFORM_PARAMS:
        aug = build_single_transform(name, seed=seed)
        rows.append(name.replace("_", " "))
        grid.extend(to_display(aug(original)) for _ in range(draws))

    full = build_full_pipeline(config_path, seed=seed)
    rows.append("FULL PIPELINE")
    grid.extend(to_display(full(original)) for _ in range(draws))

    fig, axes = plt.subplots(len(rows), draws, figsize=(2.6 * draws, 2.6 * len(rows)))
    if draws == 1:
        axes = axes.reshape(-1, 1)
    for i, label in enumerate(rows):
        for j in range(draws):
            ax = axes[i, j]
            ax.imshow(grid[i * draws + j], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=8, va="center")
            if i == 0:
                ax.set_title(f"draw {j + 1}", fontsize=8)

    stem = Path(image_path).stem[-60:]  # the full names are very long
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Outputs:")[0].strip())
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES,
                        help="holographic PNG files to visualize (one figure each)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output directory (default: ./plots/augmentation_effects)")
    parser.add_argument("--draws", type=int, default=3,
                        help="random draws per transform (default: 3)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument("--domain", choices=["tensor", "pil"], default="tensor",
                        help="image domain: tensor = float [0,1] (minmax reader), "
                             "pil = 8-bit (legacy reader); default tensor")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="stage config whose image_transforms block feeds the "
                             "'full pipeline' row")
    args = parser.parse_args()

    missing = [p for p in args.images if not Path(p).exists()]
    if missing:
        sys.exit("Image files not found:\n  " + "\n  ".join(missing) +
                 "\nPass your own images with --images.")

    for p in args.images:
        out = plot_image(p, domain=args.domain, draws=args.draws, seed=args.seed,
                         config_path=Path(args.config), out_dir=Path(args.out))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
