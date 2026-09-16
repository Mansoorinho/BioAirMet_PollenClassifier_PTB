'''
@file    :   datasets_cls.py
@create date : 2026-01-25 10:07:49
@modify date 2026-04-29 15:09:00
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This file contains classes to handle dataset objects for both stages (SSL and classification).
    It includes the HDF5Dataset class, which is used to load and preprocess the dataset for both stages.
    The dataset class is designed to be flexible and efficient, with support for multi-process loading and various augmentation strategies.
    It also includes a new image reader that preserves the 16-bit nature of the holographic images while applying robust normalization techniques suitable for device-agnostic learning.    
    There are several classes including:
    - LegacyMinMaxImageReader: A flexible image reader that applies min-max normalization to 8-bit, with optional augmentation.
    - New_Image_reader: A new image reader that applies various normalization methods while preserving 16-bit precision, with optional augmentation.
    - ImageAugmentation_Generic (see data/augmentation.py): the image augmentation
      pipeline used by both stages, with two modes: legacy mode (legacy_img_aug
      True) runs a fixed hard-coded per-stage pipeline (SSL / classification);
      custom mode (legacy_img_aug False) builds the pipeline from the
      image_transforms config block, optionally sharing the flip/rotation
      randomness across the two views of a pair
      (parallel_flipping_rotation). The active set is reported in the training log.
    - FluorescenceAugmentation (see data/augmentation.py): the config-driven
      fluorescence-spectrum augmentation pipeline (a single probability gate over
      'pca_jitter' or 'covariance_style_augmentation').
    - HDF5Dataset: A base dataset class for loading data from HDF5 files, optimized for multi-process loading (SWMR) and holographic data normalization, with support for the image and fluorescence augmentation pipelines.
    - Stage1Dataset: A dataset class for self-supervised learning (SSL) that inherits from HDF5Dataset, designed to return pairs of augmented images and their corresponding fluorescence data for contrastive learning.
    - Stage2Dataset: A dataset class for supervised classification that inherits from HDF5Dataset, designed to return pairs of augmented images, their corresponding fluorescence data, and labels for classification tasks.
    - ValidationDataset_Unlabeled: A dataset class for validation that inherits from HDF5Dataset or raw directory paths, designed to return data for evaluating model performance on unlabeled or newly acquired data.
    - build_dataset_from_config: A utility function that orchestrates dataset creation, utilizing build_ssl_dataset and build_classification_dataset to handle subsetting, splitting, and shuffling logic.
    ]
'''


import os
import ast

from pathlib import Path
from torchvision import transforms
import h5py
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from PIL import Image
from typing import Optional, Dict, Union, Tuple
from .mean_std_cov_data import unlabeled_data_stats
from .augmentation import (
    build_fluorescence_augmentation,
    ImageAugmentation_Generic,
)
from easydict import EasyDict  # Import EasyDict
import torchvision.transforms.functional as TF  # Standardize as TF
import threading
import atexit
import logging
from .clean_unlabeled_data import (get_directories_and_files,
                                  get_event_name_fast,
                                  get_df,
                                  process_events_parallel)

logger = logging.getLogger(__name__)

_HDF5_HANDLES = {}  # Global registry of handles per worker
_HDF5_LOCK = threading.Lock()  # Thread-safe handle management


def _cleanup_hdf5_handles():
    """Global cleanup on process exit."""
    with _HDF5_LOCK:
        for path, handle in _HDF5_HANDLES.items():
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        _HDF5_HANDLES.clear()


# Register cleanup on exit
atexit.register(_cleanup_hdf5_handles)



# ---------------------------------------------------------------------------
# Image readers / normalisation
# ---------------------------------------------------------------------------
# The BioAirMet corpus is 16-bit grayscale PNGs with a bright (white)
# background.  Whatever the reader, the network always sees a float tensor in
# [0, 1] ('legacy' additionally passes through an 8-bit PIL image, which is what
# its augmentation pipeline operates on).  Three readers are supported:
#
#   'legacy'  LegacyMinMaxImageReader: per-image min-max -> uint8 PIL 'L'
#             -> augmentation -> Resize + ToTensor -> [0,1] -> optional Normalize
#   'minmax'  per-image min-max  -> float32 (H, W) in [0, 1]
#   'global'  divide by the file's full scale (16-bit -> /65535, 8-bit -> /255)
#
# In all three, the background of a typical image sits at the TOP of the range
# (white ~ 1.0 / 255), which is why the augmentation fills rotated corners with
# white (see ``resolve_fill_spec`` in data/augmentation.py).
#
# The experimental readers that used to live here ('percentile',
# 'percentile_and_standardize', 'robust_zscore', 'background_adaptive') were
# removed: no configuration in use selects them, and their output ranges are not
# [0, 1], which silently breaks both the white-fill assumption above and the
# scale the pretrained weights were trained on.
SUPPORTED_IMG_READER_TYPES = ('legacy', 'minmax', 'global')
_REMOVED_IMG_READER_TYPES = ('percentile', 'percentile_and_standardize',
                             'robust_zscore', 'background_adaptive')

_warned_unexpected_mode = set()


def normalize_img_reader_type(name) -> str:
    """Validate and normalise a ``data.img_reader_type`` value.

    Accepts (case-insensitively) the supported names plus two compatibility
    aliases:
      * ``none`` / ``raw`` -> ``global``  (the inference CLI used to offer
        ``none``, which fell through to an all-zero image).
      * falsy / missing    -> ``legacy``  (the documented default).

    Raises
    ------
    ValueError
        For unknown names, and with a migration hint for the removed readers.
    """
    if name is None or (isinstance(name, str) and not name.strip()):
        return 'legacy'
    key = str(name).strip().lower()
    if key in ('none', 'raw', 'no_norm', 'nonormalize'):
        return 'global'
    if key in _REMOVED_IMG_READER_TYPES:
        raise ValueError(
            f"img_reader_type='{name}' was removed. Supported readers: "
            f"{', '.join(SUPPORTED_IMG_READER_TYPES)}. '{key}' produced output "
            "outside [0, 1]; use 'minmax' (per-image contrast stretch) or "
            "'global' (absolute full-scale division). Note that switching "
            "readers also changes the input distribution, so it MUST match the "
            "reader the SSL weights were trained with."
        )
    if key not in SUPPORTED_IMG_READER_TYPES:
        raise ValueError(
            f"Unknown img_reader_type='{name}'. "
            f"Supported: {', '.join(SUPPORTED_IMG_READER_TYPES)}."
        )
    return key


def read_image_norm_new(img_path: str, method: str = 'global',
                        img_size: int = 200) -> np.ndarray:
    """Read a holographic image and normalise it to float32 in [0.0, 1.0].

    Designed for holographic images with a white/bright background and dark
    particles with interference fringes.

    Parameters
    ----------
    img_path : str
        Path to an 8-bit or 16-bit grayscale PNG.
    method : {'minmax', 'global'}
        ``minmax``  : per-image contrast stretch ``(x - min) / (max - min)``.
        ``global``  : absolute division by the file's full scale: 65535 for
        16-bit modes (``I;16``, ``I;16B``, ``I;16L``, ``I``) and 255 for 8-bit
        (``L``, ``1``, ``P``).  This preserves the absolute intensity, so a
        white background stays white.
    img_size : int
        Size used for the zero image returned on read errors, so that the
        sample still collates with the rest of the batch.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 array in [0, 1].  On failure a zero ``(img_size,
        img_size)`` array is returned and a warning is printed (never a
        3-channel or 4-D array; ``New_Image_reader`` adds the channel dim).
    """
    method = normalize_img_reader_type(method)
    if method == 'legacy':
        # 'legacy' is a PIL/uint8 pipeline owned by LegacyMinMaxImageReader.
        method = 'minmax'

    try:
        with Image.open(img_path) as pil_img:
            mode = pil_img.mode
            img_array = np.array(pil_img, dtype=np.float32)

        # Unexpected multi-channel input: average the channels (same convention
        # as LegacyMinMaxImageReader) instead of silently dropping all but the
        # first one.
        if img_array.ndim == 3:
            print(f"Warning: {img_path} has {img_array.shape[-1]} channels "
                  f"(PIL mode '{mode}'); averaging them into one.")
            img_array = img_array.mean(axis=-1)

        # ---------------------------------------------------------
        if method == 'minmax':
            min_val, max_val = float(img_array.min()), float(img_array.max())
            range_val = max_val - min_val
            if range_val > 1e-5:
                img_array = (img_array - min_val) / range_val
            else:
                # Flat image (no particle) - return uniform gray
                img_array = np.ones_like(img_array) * 0.5

        # ---------------------------------------------------------
        elif method == 'global':
            # 8-bit PNGs are 'L' (or '1'/'P'); everything else in this corpus is
            # 16-bit ('I;16' and its byte-order variants, or 'I').  'F' is
            # already float and is scaled by its own max to stay in [0, 1].
            if mode in ('L', '1', 'P'):
                img_array = img_array / 255.0
            elif mode == 'F':
                max_val = float(img_array.max())
                img_array = img_array / max_val if max_val > 1e-5 else img_array * 0.0
            else:
                if mode not in ('I;16', 'I;16B', 'I;16L', 'I') and mode not in _warned_unexpected_mode:
                    _warned_unexpected_mode.add(mode)
                    print(f"Warning: unexpected PIL mode '{mode}' for {img_path}; "
                          "assuming 16-bit full scale (65535).")
                img_array = img_array / 65535.0
            img_array = np.clip(img_array, 0.0, 1.0)

        else:  # pragma: no cover - normalize_img_reader_type() guards this
            raise ValueError(f"Unknown normalization method: {method}")

        # Validate before conversion to prevent NaN propagation
        if not np.isfinite(img_array).all():
            print(f"Warning: NaN/Inf detected in normalized image {img_path}, replacing with zeros")
            img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)

        return img_array.astype(np.float32, copy=False)

    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        # 2-D (H, W) on purpose: New_Image_reader adds the channel dimension.
        return np.zeros((int(img_size), int(img_size)), dtype=np.float32)



class _BaseImageReader:
    """Shared base for image readers: one common contract for using augmentation.

    A reader loads 1 or 2 images into its own representation (PIL or tensor),
    then hands them to the optional ``augmentation`` callable through the single
    ``_apply_augmentation`` path below:

    * gated by ``p`` (applied with probability ``p``; set 0.0 for clean data),
    * called with the loaded images (1 or 2), whatever type the reader produces,
    * works with any callable: augmentations that declare
      ``supports_pairs = True`` (``ImageAugmentation_Generic``) receive the two
      views of a pair in ONE call (so shared pair geometry works); every other
      callable is applied to each image independently.
    """

    def __init__(self, p: float = 0.5, augmentation=None, img_size: int = 200,
                 img_mean: Optional[float] = None, img_std: Optional[float] = None,
                 stitch: bool = False):
        self.p = p
        self.augmentation = augmentation
        self.img_size = img_size
        self.img_mean = img_mean
        self.img_std = img_std
        # When True and two views are read, they are combined into one image
        # (config `data.stitch_images`).
        self.stitch = stitch

    def _apply_augmentation(self, *imgs) -> Tuple:
        """Apply ``self.augmentation`` (gated by ``self.p``) to 1 or 2 images.

        Augmentations that declare ``supports_pairs = True``
        (``ImageAugmentation_Generic``) receive a 2-image pair in ONE call so
        they can share the flip/rotation randomness across the views
        (``parallel_flipping_rotation``).  Every other callable is applied to
        each image independently, so any single-input callable works
        (torchvision transforms, ``Compose``, plain functions).  The gate ``p``
        is sampled once per ``__call__``, so an image pair is either augmented
        together or not at all.

        Returns a tuple of the (possibly augmented) images, same length as input.
        """
        if self.augmentation is None:
            return imgs
        if torch.rand(1).item() < self.p:
            if len(imgs) == 2 and getattr(self.augmentation, "supports_pairs", False):
                return tuple(self.augmentation(list(imgs)))
            return tuple(self.augmentation(img) for img in imgs)
        return imgs


class LegacyMinMaxImageReader(_BaseImageReader):
    """
    Flexible image reader for 1 or 2 grayscale images with minmax normalization to 8-bit.
    Supports augmentation and optional normalization.
    """

    def __init__(self, p: float = 0.5, augmentation=None, img_size: int = 200,
                 img_mean: Optional[float] = None, img_std: Optional[float] = None,
                 stitch: bool = False):
        super().__init__(p=p, augmentation=augmentation, img_size=img_size,
                         img_mean=img_mean, img_std=img_std, stitch=stitch)

        # Base transform: PIL -> tensor
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # Add normalization if provided
        if img_mean is not None and img_std is not None:
            self.base_transform = transforms.Compose([
                self.base_transform,
                transforms.Normalize(mean=[img_mean], std=[img_std])
            ])

    def _read_image(self, img_path: str) -> Image.Image:
        """Minmax normalize 16-bit -> 8-bit grayscale PIL Image."""
        try:
            pil_img = Image.open(img_path)
            img_array = np.array(pil_img, dtype=np.float32)

            # Ensure grayscale
            if img_array.ndim > 2:
                img_array = np.mean(img_array, axis=-1)

            # Minmax normalization
            min_val, max_val = img_array.min(), img_array.max()
            if max_val > min_val:
                normalized = (img_array - min_val) / (max_val - min_val)
                img_array = (normalized * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)

            normalized_img = Image.fromarray(img_array, mode='L')
            pil_img.close()
            return normalized_img

        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")
        except Exception as e:
            raise ValueError(f"Failed to process {img_path}: {e}")

    def __call__(self, img_input: Union[str, Tuple[str, str]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process 1 or 2 images.

        Args:
            img_input: str path OR tuple of 2 paths

        Returns:
            Single tensor OR tuple of 2 tensors
        """
        # Single image
        if isinstance(img_input, str):
            img = self._apply_augmentation(self._read_image(img_input))[0]
            return self.base_transform(img) # type: ignore

        # Two images (e.g., input/target pair)
        elif len(img_input) == 2:
            img1, img2 = self._apply_augmentation(self._read_image(img_input[0]),
                                                  self._read_image(img_input[1]))
            t1, t2 = self.base_transform(img1), self.base_transform(img2) # type: ignore
            # Optionally stitch the two views side-by-side into a single image.
            return torch.cat([t1, t2], dim=2) if self.stitch else (t1, t2)

        raise TypeError(f"Expected str or (str,str), got {type(img_input)}")

class New_Image_reader(_BaseImageReader):
    """Tensor reader: normalises 8/16-bit holographic images to float32 [0, 1].

    ``method`` selects the normalisation applied by :func:`read_image_norm_new`
    ('minmax' = per-image contrast stretch, 'global' = absolute full-scale
    division; 'none' is accepted as an alias of 'global').  The output is always
    a ``(1, H, W)`` float tensor in [0, 1] - i.e. exactly what PyTorch models
    expect, with the white background at the top of the range.

    Unlike :class:`LegacyMinMaxImageReader` this reader did not resize at all
    (it assumed the files already have ``data.image_size`` pixels), so an image
    of a different size produced a collate error.  Such images are now resized
    bilinearly and a one-time warning is printed.
    """

    def __init__(self, p: float = 0.5,
                 augmentation=None, img_size: int = 200,
                 img_mean: Optional[float] = None, img_std: Optional[float] = None,
                 method: str = 'minmax', stitch: bool = False):
        super().__init__(p=p, augmentation=augmentation, img_size=img_size,
                         img_mean=img_mean, img_std=img_std, stitch=stitch)
        self.method = normalize_img_reader_type(method)
        self._size_warned = False

    def _to_tensor(self, img_path: str) -> torch.Tensor:
        """Read + normalize an image to a (1, img_size, img_size) float tensor."""
        img_array = read_image_norm_new(img_path, method=self.method,
                                        img_size=self.img_size)
        tensor = torch.from_numpy(
            np.ascontiguousarray(img_array, dtype=np.float32)
        ).unsqueeze(0)

        if tensor.shape[-2] != self.img_size or tensor.shape[-1] != self.img_size:
            if not self._size_warned:
                print(
                    f"Warning: {img_path} is {tuple(tensor.shape[-2:])} "
                    f"but data.image_size={self.img_size}; resizing (warned once per loader)."
                )
                self._size_warned = True
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        return tensor

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ``data.image_normalization`` (mean/std) AFTER augmentation.

        :class:`LegacyMinMaxImageReader` has always done this (``ToTensor`` then
        ``Normalize``), but this reader accepted ``img_mean``/``img_std`` and
        silently ignored them, so enabling image normalization changed nothing
        for 'minmax'/'global'.  It is applied last, so the augmentations (and their
        white fill) always work on the [0, 1] image.
        """
        if self.img_mean is None or self.img_std is None:
            return tensor
        return (tensor - float(self.img_mean)) / float(self.img_std)

    def __call__(self, img_input: Union[str, Tuple[str, str]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Single image
        if isinstance(img_input, str):
            img_tensor = self._apply_augmentation(self._to_tensor(img_input))[0]
            # Float ops (blur/jitter) can drift a few ulps past [0, 1]; the
            # clamp keeps the tensor domain as tight as the 8-bit PIL path.
            img_tensor = img_tensor.clamp(0.0, 1.0)
            return self._normalize(img_tensor) # type: ignore

        # Two images (e.g., input/target pair)
        elif len(img_input) == 2:
            img1_tensor, img2_tensor = self._apply_augmentation(self._to_tensor(img_input[0]),
                                                                self._to_tensor(img_input[1]))
            img1_tensor = img1_tensor.clamp(0.0, 1.0)
            img2_tensor = img2_tensor.clamp(0.0, 1.0)
            img1_tensor, img2_tensor = self._normalize(img1_tensor), self._normalize(img2_tensor)
            # Optionally stitch the two views side-by-side into a single image.
            return (torch.cat([img1_tensor, img2_tensor], dim=2) if self.stitch
                    else (img1_tensor, img2_tensor)) # type: ignore

        raise TypeError(f"Expected str or (str,str), got {type(img_input)}")


class HDF5Dataset(Dataset):
    """
    Base dataset class for loading data from HDF5 files.
    Optimized for multi-process loading (SWMR) and holographic data normalization.
    """

    def __init__(
        self,
        hdf5_path: str,
        image_key: str = 'images',
        fluorescence_key: str = 'fluorescence',
        label_key: str = 'labels',  # or 'category_num'
        fl_aug_type: Optional[str] = 'pca_jitter',
        fluo_pca_std: float = 0.1,  # Jitter magnitude (std) for the 'pca_jitter' fl augmentation
        return_labels: bool = False,
        unlabeled_path: Optional[str] = None,
        fluo_aug_prob: float = 0.0,
        img_aug_prob: float = 0.25,  # Probability of applying image augmentation
        legacy_augmentation: bool = True,  # Whether to use legacy augmentations
        legacy_stage: str = "ssl",  # Which legacy preset to use when legacy_augmentation is True ('ssl' | 'classification')
        image_transforms: Optional[dict] = None,  # Per-transform image augmentation (true/false on/off)
        img_reader_type: str = 'legacy',  # Whether to use the legacy image augmentation pipeline
        train_stats: Optional[dict] = None,  # specific for FL augmentation
        unlabeled_stats: Optional[dict] = None,
        skip_validation: bool = False,  # only on the unlabeled dataset
        # Image-reader options (config `data.image_size`, `data.image_normalization`,
        # `data.stitch_images`).
        image_size: int = 200,
        img_mean: Optional[float] = None,
        img_std: Optional[float] = None,
        stitch_images: bool = False,
        seed: Optional[int] = None,
        **unexpected,
    ):
        """
        Initialize HDF5 dataset.

        Args:
            hdf5_path: Path to HDF5 file
            image_key: Key for images in HDF5 file
            fluorescence_key: Key for fluorescence data
            label_key: Key for labels (if applicable)
            normalization: Normalization method to use for images, since they are 16-bit
            fl_aug_type: Fluorescence augmentation type: None (disabled) |
                'pca_jitter' (alias: 'pca') | 'covariance_style_augmentation'
            fluo_pca_std: Jitter magnitude (std) used by 'pca_jitter' (default 0.1)
            return_labels: Whether to return labels
            fluo_aug_prob: Probability of applying fluorescence augmentation
            train_stats: Training statistics for augmentation
            unlabeled_stats: Unlabeled data statistics for augmentation
            **unexpected: Any extra keyword argument is rejected with a clear error.
        """
        if unexpected:
            _valid = [
                'hdf5_path', 'image_key', 'fluorescence_key', 'label_key', 'fl_aug_type', 'fluo_pca_std',
                'return_labels', 'unlabeled_path', 'fluo_aug_prob', 'img_aug_prob',
                'legacy_augmentation', 'legacy_stage', 'image_transforms', 'img_reader_type', 'train_stats', 'unlabeled_stats',
                'skip_validation',
                'image_size', 'img_mean', 'img_std', 'stitch_images', 'seed',
            ]
            raise TypeError(
                f"HDF5Dataset.__init__() got unexpected keyword argument(s): {sorted(unexpected)}. "
                f"Valid arguments are: {_valid}. "
                f"If you are subclassing, pop your own parameters before calling super().__init__()."
            )
        self.hdf5_path = hdf5_path
        # self.cat_map_path = cat_map_path
        self.image_key = image_key
        self.fluorescence_key = fluorescence_key
        self.label_key = label_key
        # self.normalization = normalization
        self.img_aug_prob = img_aug_prob
        self.unlabeled_path = unlabeled_path
        self.skip_validation = skip_validation
        # Config-driven image augmentation. The legacy flag is AUTHORITATIVE:
        # ``legacy_img_aug: True`` always runs the fixed hard-coded legacy
        # pipeline and the per-transform entries of the ``image_transforms``
        # config block are ignored; those entries only apply in custom mode
        # (``legacy_img_aug: False``), where a missing/empty block means
        # identity (no augmentation).  The block's ``fill`` setting applies to
        # BOTH modes (pixels exposed by rotation/translation; cutout in custom
        # mode).  The legacy pipeline is STAGE-SPECIFIC (``legacy_stage``
        # selects between the SSL and the classification pipeline); in custom
        # mode the optional ``parallel_flipping_rotation`` flag shares the
        # flip/rotation randomness across the two views of a pair.
        self.legacy_stage = legacy_stage
        self.image_transforms = image_transforms or {}
        self.legacy_img_aug = bool(legacy_augmentation)
        # One pipeline serves both single-image (SSL) and image-pair (classification);
        # __call__ handles the number of inputs.
        self.transform = ImageAugmentation_Generic(self.image_transforms,
                                                   legacy=legacy_augmentation,
                                                   stage=legacy_stage,
                                                   gate_prob=img_aug_prob,
                                                   seed=seed)
                
        # Fail fast on a typo/removed reader name instead of deep inside the loader
        img_reader_type = normalize_img_reader_type(img_reader_type)
        if img_reader_type == 'legacy':
            self.image_reader = LegacyMinMaxImageReader(
                p=img_aug_prob, augmentation=self.transform, img_size=image_size,
                img_mean=img_mean, img_std=img_std, stitch=stitch_images)
        else:
            self.image_reader = New_Image_reader(
                p=img_aug_prob, augmentation=self.transform, img_size=image_size,
                method=img_reader_type, img_mean=img_mean, img_std=img_std,
                stitch=stitch_images)
        
        self.return_labels = return_labels
        self.train_stats = train_stats
        self.unlabeled_stats = unlabeled_stats
        self.fluo_pca_std = fluo_pca_std
        # 'pca' is a deprecated alias of 'pca_jitter' (configs written against the
        # old name keep working); the alias is resolved inside the pipeline below.
        if fl_aug_type == 'pca':
            logger.warning(
                "fl_aug_type='pca' is a deprecated alias of 'pca_jitter'; "
                "please update your config.")
        # Single config-driven fluorescence pipeline: one probability gate + one
        # strategy. The dataset exposes fluo_aug_prob / fl_aug_type /
        # fluo_augmentor as thin properties backed by this object (see below), so
        # existing code, logging and tests keep working unchanged.
        self.fluorescence_augmentation = build_fluorescence_augmentation(
            style=fl_aug_type, prob=fluo_aug_prob, std=fluo_pca_std,
            styles=unlabeled_stats)
        self.epoch = 0

        # --- HDF5 Initialization Logic ---
        # Calculate length ONCE in main process to prevent thundering herd
        try:
            with h5py.File(hdf5_path, 'r') as f:
                self.length = len(f[image_key])
        except (IOError, KeyError) as e:
            raise IOError(
                f"Could not read {image_key} from {hdf5_path}. Error: {e}")

        # Lazy Loading variables
        self.hf = None
        self._worker_pid = None
        self._worker_id = None

        self._last_epoch = -1
        self._handle_lock = threading.Lock()  # Per-dataset lock for safety

    def __len__(self) -> int:
        return self.length

    def class_counts(self, num_classes: int = None, raw: bool = False) -> np.ndarray:
        """Per-class sample counts for the label column.

        Reads the label array from the HDF5 file once (labels only, no
        images or spectra are touched) and applies the pre-filter
        (``self._valid_indices``, set when a category map or the configured
        ``num_classes`` range restricts the dataset) so the counts match
        exactly the samples the dataset yields. Used to derive imbalance
        weights (e.g. the focal-loss alpha) and to report class coverage.

        Args:
            num_classes: If given, the returned vector is trimmed/padded to
                exactly this length (unseen class indices count as 0).
            raw: If True, ignore the pre-filter and count every sample in the
                HDF5 file (i.e. what was excluded by the category map /
                num_classes range is included).

        Returns:
            np.ndarray of int64 with one entry per class.
        """
        with h5py.File(self.hdf5_path, 'r') as hf:
            raw_labels = np.array(hf[self.label_key][:])
        try:
            labels = raw_labels.astype(int)
        except (ValueError, TypeError):
            def _to_int(x):
                if isinstance(x, bytes):
                    x = x.decode('utf-8')
                return int(x)
            labels = np.array([_to_int(x) for x in raw_labels])
        valid_indices = None if raw else getattr(self, '_valid_indices', None)
        if valid_indices is not None:
            labels = labels[valid_indices]

        n = int(num_classes) if num_classes else 0
        if len(labels) == 0:
            return np.zeros(n, dtype=np.int64)
        minlength = n if n else int(labels.max()) + 1
        counts = np.bincount(labels, minlength=minlength).astype(np.int64)
        if n:
            if len(counts) < n:
                counts = np.pad(counts, (0, n - len(counts)))
            else:
                counts = counts[:n]
        return counts

    def _get_hdf5_handle(self):
        """
        Get thread-safe, persistent HDF5 handle with proper lifecycle.
        Uses global registry to prevent duplicate handles per worker.
        """
        current_pid = os.getpid()
        worker_info = torch.utils.data.get_worker_info()

        # Create unique key for this worker
        worker_key = f"{self.hdf5_path}_{current_pid}_{worker_info.id if worker_info else 'main'}"

        with _HDF5_LOCK:
            # Get or create handle for this worker
            if worker_key not in _HDF5_HANDLES or _HDF5_HANDLES[worker_key] is None:
                # Stagger opening to prevent thundering herd
                if worker_info is not None:
                    stagger_delay = random.uniform(
                        0.0, min(0.05 * worker_info.id, 1.0))
                    time.sleep(stagger_delay)

                try:
                    h = h5py.File(
                        self.hdf5_path,
                        'r',
                        # libver='latest',
                        # swmr=True,
                        # 1MB cache PER WORKER (reasonable)
                        # rdcc_nbytes=1024*1024,
                        # rdcc_nslots=521  # Hash table size for cache
                    )
                    _HDF5_HANDLES[worker_key] = h
                except Exception as e:
                    raise IOError(
                        f"Failed to open HDF5 file {self.hdf5_path}: {e}")

            return _HDF5_HANDLES[worker_key]

    def _close_hdf5(self):
        """
        Close handle if held by this instance.
        SAFE: Uses global registry to avoid double-closes.
        """
        # Guard: path-mode datasets have no HDF5 handle to close
        if not getattr(self, 'hdf5_path', None):
            return
        current_pid = os.getpid()
        worker_info = torch.utils.data.get_worker_info()
        worker_key = f"{self.hdf5_path}_{current_pid}_{worker_info.id if worker_info else 'main'}"

        with _HDF5_LOCK:
            if worker_key in _HDF5_HANDLES and _HDF5_HANDLES[worker_key] is not None:
                try:
                    _HDF5_HANDLES[worker_key].close()
                except Exception:
                    pass
                _HDF5_HANDLES[worker_key] = None

    def __del__(self):
        self._close_hdf5()
 
    # --- fluorescence augmentation accessors (backed by one pipeline) --------
    # A single ``FluorescenceAugmentation`` object owns the strategy + the single
    # probability gate. These properties keep the historical attribute names
    # (fluo_aug_prob / fl_aug_type / fluo_augmentor) working for callers, logging
    # and tests, while the logic lives in one place.
    @property
    def fluo_aug_prob(self) -> float:
        """The single fluorescence augmentation probability gate."""
        return self.fluorescence_augmentation.prob

    @fluo_aug_prob.setter
    def fluo_aug_prob(self, value) -> None:
        self.fluorescence_augmentation.prob = float(value)

    @property
    def fl_aug_type(self):
        """The active fluorescence strategy (``None`` disables augmentation)."""
        return self.fluorescence_augmentation.style

    @fl_aug_type.setter
    def fl_aug_type(self, value) -> None:
        self.fluorescence_augmentation.set_style(value)

    @property
    def fluo_augmentor(self):
        """The PCA jitter augmentor (``None`` unless the strategy is ``pca_jitter``)."""
        return self.fluorescence_augmentation.pca_jitter

    def _process_fluorescence(self, spectra_values: np.ndarray) -> torch.Tensor:
        """Process a fluorescence spectrum with optional augmentation.

        Keeps 13 channels (0-4, 6-9, 11-14), dropping indices 5 and 10 which are
        always zero. A raw 15-dim spectrum is reduced to 13-dim to match
        ``fluorescence_tower.input_dim`` (13); an already-reduced 13-dim spectrum
        passes through unchanged.

        The augmentation itself (strategy + the single ``fluo_aug_prob`` gate) is
        owned by :attr:`fluorescence_augmentation`; this method only handles the
        dimension guard and the 15 -> 13 reduction.

        Args:
            spectra_values (np.ndarray | torch.Tensor): raw fluorescence spectrum.
        Returns:
            torch.Tensor: reduced spectrum of shape ``(13,)``.
        """
        # Normalise to a 1-D float32 numpy array for consistent downstream handling.
        if isinstance(spectra_values, torch.Tensor):
            spectra_values = spectra_values.detach().cpu().numpy()
        spectra_values = np.asarray(spectra_values, dtype=np.float32).ravel()

        n = spectra_values.shape[0]
        if n not in (13, 15):
            raise ValueError(
                f"Unexpected raw fluorescence dimension {n}: BioAirMet expects a 15-dim "
                f"spectrum (reduced to 13-dim by dropping the zero channels at indices 5 "
                f"and 10) or an already-reduced 13-dim feature matching "
                f"fluorescence_tower.input_dim. Check the HDF5 'fluorescence' column."
            )

        # A raw 15-dim spectrum may be augmented (on 15-dim) before reduction; a
        # pre-reduced 13-dim spectrum is passed through unchanged (the strategies
        # are defined for the 15-dim space, so they only apply to the raw case).
        if n == 15:
            spectra_values = self.fluorescence_augmentation(spectra_values)
            # Strategies return a torch.Tensor; bring it back to numpy for indexing.
            if isinstance(spectra_values, torch.Tensor):
                spectra_values = spectra_values.detach().cpu().numpy()
            spectra_values = np.asarray(spectra_values, dtype=np.float32)
            # Reduce to the model-facing dimension: drop the always-zero channels 5 & 10.
            keep = [i for i in range(15) if i not in (5, 10)]
            spectra_values = spectra_values[keep]

        return torch.as_tensor(spectra_values, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("Use Stage1Dataset or Stage2Dataset")

    def set_epoch(self, epoch: int):
        """
        Set the current epoch.
        """
        self.epoch = epoch
        # Reset handle periodically to prevent stale connections
        # if (epoch > 0 and epoch % 10 == 0):
        #     self._close_hdf5()  # Will be reopened on next access

class Stage1Dataset(HDF5Dataset):
    """
    Upgraded Dataset for Stage 1: Self-Supervised Learning.
    Returns: {'image': Tensor(C,H,W), 'fluorescence': Tensor(13,)}
    """

    def __init__(self, **kwargs):
        # Default keys for SSL if not provided
        if 'image_key' not in kwargs:
            kwargs['image_key'] = 'images'
        if 'fluorescence_key' not in kwargs:
            kwargs['fluorescence_key'] = 'relative_spectra'
        if 'label_key' not in kwargs:
            kwargs['label_key'] = None  # Not used in SSL
        if 'legacy_augmentation' not in kwargs:
            kwargs['legacy_augmentation'] = True  # Use legacy augmentations for SSL by default
        kwargs.setdefault('legacy_stage', 'ssl')  # SSL legacy: jitter+blur group + flips
        super().__init__(return_labels=False, **kwargs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._get_hdf5_handle()

        # Load Raw Data (Paths from HDF5)
        image_path = f[self.image_key][idx]
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        else:
            print(f"Warning: Expected bytes for image path, got {type(image_path)}. Attempting to convert.")

        fluo_data = f[self.fluorescence_key][idx]
        
        image_tensor = self.image_reader(image_path)

        # Process Fluorescence (Slice + Augment)
        fluo_tensor = self._process_fluorescence(fluo_data)

        return {
            'image': image_tensor,
            'fluorescence': fluo_tensor
        }


class Stage2Dataset(HDF5Dataset):
    """
    Upgraded Dataset for Stage 2: Supervised Learning.
    Returns: {'image': Tensor(C,H,W), 'label': int}
    """

    def __init__(self, cat_map_path=None, num_classes=None, **kwargs):
        # Default keys for Stage 2 if not provided
        if 'image_key' not in kwargs:
            kwargs['image_key'] = 'images'
        if 'fluorescence_key' not in kwargs:
            kwargs['fluorescence_key'] = 'relative_spectra'
        if 'label_key' not in kwargs:
            kwargs['label_key'] = 'category_num'
        # Stage 2 (classification) legacy augmentation is stage-specific: when
        # legacy_img_aug is True it uses the classification legacy pipeline
        # (flips + rotation ±25° + translation 5% + jitter/blur) instead of
        # the SSL one (atomic jitter+blur group + flips).
        kwargs.setdefault('legacy_stage', 'classification')
        if kwargs.get('stitch_images', False):
            raise ValueError(
                "data.stitch_images=True is not supported for two-image "
                "classification data (Stage2Dataset). Stitching combines the two "
                "views into one image, which does not match the (view_a, view_b) "
                "classification model forward pass. Set stitch_images=False for stage 2.")
        self.hf = None
        self.cat_map_path = cat_map_path
        # self.unlabeled_stats = unlabeled_data_stats
        # self.transform = TieredHolographicAugmentation(stage="supervised")

        super().__init__(return_labels=True, **kwargs)

        self._valid_indices = None
        if self.cat_map_path:
            try:
                mapping = self._get_category_mapping()
                if mapping:
                    # mapping: name->number  -> we want allowed numeric labels
                    allowed = set(mapping.values())
                    try:
                        with h5py.File(self.hdf5_path, 'r') as hf:
                            labels = hf[self.label_key][:]
                            # Convert bytes/strings to ints if necessary
                            labels_arr = np.array(labels)
                            # Try to coerce to int dtype
                            try:
                                labels_int = labels_arr.astype(int)
                            except Exception:
                                # fallback: decode bytes then int
                                labels_int = np.array([int(x) for x in labels_arr])

                            valid_mask = [int(l) in allowed for l in labels_int]
                            self._valid_indices = [i for i, v in enumerate(valid_mask) if v]
                            self.length = len(self._valid_indices)
                    except Exception:
                        # If reading HDF5 labels fails, leave dataset unfiltered
                        self._valid_indices = None
            except Exception:
                self._valid_indices = None

        # Exclude samples whose label is outside the configured head range
        # (label >= num_classes): the classifier head cannot represent them
        # and the loss would crash with an index-out-of-bounds error. This
        # is the "skip the extra garbage classes" behaviour; the dataset
        # simply drops those samples (deterministic, identical on every
        # rank). Merged with the category-map pre-filter above (intersection).
        # ``check_label_coverage`` (called by the training worker) logs which
        # classes/samples were excluded and which configured classes have no
        # samples at all.
        if num_classes is not None:
            try:
                n_classes = int(num_classes)
                with h5py.File(self.hdf5_path, 'r') as hf:
                    labels_raw = np.array(hf[self.label_key][:])
                try:
                    labels_raw = labels_raw.astype(int)
                except (ValueError, TypeError):
                    def _to_int(x):
                        if isinstance(x, bytes):
                            x = x.decode('utf-8')
                        return int(x)
                    labels_raw = np.array([_to_int(x) for x in labels_raw])
                in_range = [i for i, l in enumerate(labels_raw) if int(l) < n_classes]
                if self._valid_indices is not None:
                    allowed = set(self._valid_indices)
                    in_range = [i for i in in_range if i in allowed]
                self._valid_indices = in_range
                self.length = len(in_range)
            except Exception:
                # If reading the labels fails, keep the current (possibly
                # category-map-filtered) dataset unchanged.
                pass

    def _get_category_mapping(self):
        category_mapping = {}
        # Resolve a relative cat_map_path (CWD-independent) so class names are found
        # regardless of the launch directory; see _resolve_category_map_path.
        resolved_map = _resolve_category_map_path(self.cat_map_path) if self.cat_map_path else None
        if resolved_map is None or not Path(resolved_map).exists():
            return None
        try:
            with open(resolved_map, 'r') as file:
                next(file)  # Skip the first line
                for line in file:
                    line = line.strip()
                    if line:  # Ensure the line is not empty
                        key_value = line.split('\t')
                        if len(key_value) == 2:
                            category = key_value[0].strip()
                            number = int(key_value[1].strip())
                            category_mapping[category] = number
        except Exception as e:
            print(f"Warning: Could not read category mapping: {e}")
            return None

        return category_mapping if category_mapping else None

    def __getitem__(self, idx: int) -> Dict[str, Union[tuple, torch.Tensor, int]]:
        f = self._get_hdf5_handle()
        # Map dataset-local index to the underlying HDF5 index if we pre-filtered
        real_idx = self._valid_indices[idx] if (hasattr(self, '_valid_indices') and self._valid_indices is not None) else idx

        # Load Raw Data (Paths from HDF5)
        image_path = f[self.image_key][real_idx]
        # Support a few possible stored representations: bytes (encoded list/str), str, or already a list
        image_data = None
        if isinstance(image_path, bytes):
            try:
                image_data = ast.literal_eval(image_path.decode('utf-8'))
            except Exception:
                image_data = image_path.decode('utf-8')
        elif isinstance(image_path, str):
            try:
                image_data = ast.literal_eval(image_path)
            except Exception:
                image_data = image_path
        else:
            # assume it's already a Python object (list/tuple)
            image_data = image_path

        fluo_data = f[self.fluorescence_key][real_idx]

        image_tensor_0, image_tensor_1 = self.image_reader(image_data)

        fluo_tensor = self._process_fluorescence(fluo_data)
        # Process Label
        # Read and coerce label from the underlying storage
        raw_label = f[self.label_key][real_idx]
        try:
            label_val = int(raw_label)
        except Exception:
            # fallback: try decoding bytes then int
            try:
                label_val = int(raw_label.decode('utf-8'))
            except Exception:
                raise RuntimeError(f"Could not parse label at index {real_idx}: {raw_label}")

        label_tensor = torch.tensor(label_val, dtype=torch.long)
        return {
            'image': (image_tensor_0, image_tensor_1),
            'fluorescence': fluo_tensor,
            'label': label_tensor
        }


class ValidationDataset_Unlabeled(HDF5Dataset):
    """
    Dataset for Validation.
    Supports two modes:
      - HDF5 mode  : provide ``hdf5_path`` (and optionally ``unlabeled_path=None``).
      - Path mode  : provide ``unlabeled_path`` to raw event directories
                     (``hdf5_path`` can be omitted / None).
    Returns: {'image': (Tensor, Tensor), 'fluorescence': Tensor(13,), 'image_path': str}
    """

    def __init__(self, **kwargs):
        # Default keys for Validation if not provided
        if 'image_key' not in kwargs:
            kwargs['image_key'] = 'images'
        if 'fluorescence_key' not in kwargs:
            kwargs['fluorescence_key'] = 'relative_spectra'
            
        if 'min_intensity_delta' not in kwargs:
            kwargs['min_intensity_delta'] = 0.1
        if 'min_area' not in kwargs:
            kwargs['min_area'] = 500
        if 'min_solidity' not in kwargs:
            kwargs['min_solidity'] = 0.7
        if 'skip_validation' not in kwargs:
            kwargs['skip_validation'] = False

        unlabeled_path = kwargs.get('unlabeled_path', None)
        hdf5_path = kwargs.get('hdf5_path', None)

        # --- Path mode: build dataset entirely from raw event directories ---
        self._path_mode = (unlabeled_path is not None) and (hdf5_path is None)

        if self._path_mode:
            # Build lightweight image-reader / augmentor without touching HDF5.
            # We bypass super().__init__ and initialise only what we need.
            img_aug_prob = kwargs.get('img_aug_prob', 0.0)
            img_reader_type = normalize_img_reader_type(kwargs.get('img_reader_type', 'legacy'))
            fl_aug_type = kwargs.get('fl_aug_type', 'pca_jitter')
            fluo_aug_prob = 0.0
            train_stats = kwargs.get('train_stats', None)
            unlabeled_stats = kwargs.get('unlabeled_stats', None)

            self.hdf5_path = None
            self.unlabeled_path = unlabeled_path
            self.skip_validation = kwargs.get('skip_validation', False)
            self.image_key = kwargs['image_key']
            self.fluorescence_key = kwargs['fluorescence_key']
            self.return_labels = False
            # No augmentation during validation: the fluorescence gate is
            # closed (prob=0.0) and the image reader gets no augmentation.
            self.img_aug_prob = 0.0
            self.train_stats = train_stats
            self.unlabeled_stats = unlabeled_stats
            self.fluorescence_augmentation = build_fluorescence_augmentation(
                style=fl_aug_type, prob=0.0, std=0.1, styles=unlabeled_stats)
            self.epoch = 0
            self._handle_lock = threading.Lock()
            self.hf = None
            self._worker_pid = None
            self._worker_id = None
            self._last_epoch = -1

            # Image reader (no augmentation during validation)
            _rd = image_reader_kwargs(kwargs)
            if img_reader_type == 'legacy':
                self.image_reader = LegacyMinMaxImageReader(
                    p=0.0, augmentation=None, img_size=_rd['image_size'],
                    img_mean=_rd.get('img_mean'), img_std=_rd.get('img_std'),
                    stitch=_rd.get('stitch_images', False))
            else:
                self.image_reader = New_Image_reader(
                    p=0.0, augmentation=None, img_size=_rd['image_size'],
                    method=img_reader_type, img_mean=_rd.get('img_mean'),
                    img_std=_rd.get('img_std'), stitch=_rd.get('stitch_images', False))

            # --- Build DataFrame from raw path ---
            print("Starting data processing...")
            print(f"1. Getting directories and files from: {unlabeled_path}")
            data = get_directories_and_files(unlabeled_path)

            print("2. Extracting event names and image paths...")
            data_fast = get_event_name_fast(data, unlabeled_path)

            print("3. Converting to DataFrame and filtering invalid entries...")
            data_final, _df_all_events = get_df(data_fast)

            print("4. Extracting fluorescence spectra with memory-optimized parallel processing...")
            event_paths = data_final['event'].tolist()
            min_area = kwargs.get('min_area', 500)
            min_solidity = kwargs.get('min_solidity', 0.7)
            min_intensity_delta = kwargs.get('min_intensity_delta', 0.1)
            spectra_results = process_events_parallel(events=event_paths, min_area=min_area, min_solidity=min_solidity, min_intensity_delta=min_intensity_delta, skip_validation=self.skip_validation)
            data_final['relative_spectra'] = spectra_results

            print("5. Filtering out invalid spectra entries...")
            mask = data_final['relative_spectra'].apply(lambda x: not isinstance(x, int))
            num_valid_spectra = mask.sum()
            print(f"Number of valid spectra: {num_valid_spectra} out of {len(spectra_results)}")

            self.df = data_final[mask].reset_index(drop=True)
            self.length = len(self.df)
            print(f"Shape of cleaned DataFrame with valid spectra: {self.df.shape}")

        else:
            # --- HDF5 mode: standard initialisation ---
            # min_* are path-mode-only filtering thresholds (consumed by the path
            # branch above); they are not part of the base signature, so drop them
            # before super().__init__ to avoid leaking unknown keyword arguments.
            for _k in ('min_intensity_delta', 'min_area', 'min_solidity'):
                kwargs.pop(_k, None)
            # Resolve the config-level reader options (image_size,
            # image_normalization (gated on the enable flag), stitch_images)
            # into the base signature's image_size/img_mean/img_std keys so
            # ``super().__init__`` does not reject the unknown
            # ``image_normalization`` key.  Explicitly passed values win.
            _rd = image_reader_kwargs(kwargs)
            kwargs.pop('image_normalization', None)
            for _k, _v in _rd.items():
                kwargs.setdefault(_k, _v)
            super().__init__(return_labels=False, **kwargs)
            self.fluo_aug_prob = 0.0   # No augmentation during validation
            self.img_aug_prob = 0.0    # No augmentation during validation
            self.df = None

            # Also process unlabeled_path if provided alongside HDF5 (optional side-effect)
            if unlabeled_path:
                print("Starting data processing...")
                print(f"1. Getting directories and files from: {unlabeled_path}")
                data = get_directories_and_files(unlabeled_path)

                print("2. Extracting event names and image paths...")
                data_fast = get_event_name_fast(data, unlabeled_path)

                print("3. Converting to DataFrame and filtering invalid entries...")
                data_final, _df_all_events = get_df(data_fast)

                print("4. Extracting fluorescence spectra with memory-optimized parallel processing...")
                event_paths = data_final['event'].tolist()
                spectra_results = process_events_parallel(event_paths)
                data_final['relative_spectra'] = spectra_results

                print("5. Filtering out invalid spectra entries...")
                mask = data_final['relative_spectra'].apply(lambda x: not isinstance(x, int))
                num_valid_spectra = mask.sum()
                print(f"Number of valid spectra: {num_valid_spectra} out of {len(spectra_results)}")

                self.df = data_final[mask].reset_index(drop=True)
                print(f"Shape of cleaned DataFrame with valid spectra: {self.df.shape}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Union[tuple, torch.Tensor, int]]:
        if self._path_mode:
            # --- Path mode: read directly from DataFrame ---
            row = self.df.iloc[idx]
            image_path = row['images']  # list of 2 image paths
            fluo_data = np.array(row['relative_spectra'], dtype=np.float32)
            fluo_tensor = self._process_fluorescence(fluo_data)

            img_0, img_1 = self.image_reader(image_path)

            return {
                'image': (img_0, img_1),
                'fluorescence': fluo_tensor,
                'image_path': image_path[0]
            }

        else:
            # --- HDF5 mode ---
            f = self._get_hdf5_handle()

            image_path = f[self.image_key][idx]
            if isinstance(image_path, bytes):
                image_path = ast.literal_eval(image_path.decode('utf-8'))

            fluo_data = f[self.fluorescence_key][idx]
            fluo_tensor = self._process_fluorescence(fluo_data)

            img_0, img_1 = self.image_reader(image_path)

            return {
                'image': (img_0, img_1),
                'fluorescence': fluo_tensor,
                'image_path': image_path[0]
            }


def build_dataset_from_config(config: EasyDict):
    """
    Builds a dataset instance based on the provided configuration.

    Args:
        config (EasyDict): Configuration object.
        split (str): 'train' or 'val' to determine which dataset split to load.
        subset_percentage (float): Percentage of the dataset to use (0.0 to 1.0).

    Returns:
        Dataset: An instance of the appropriate dataset class.
    """
    experiment_type = config.architecture_setup.type  # type: ignore , e.g., 'classification', 'ssl'
    if experiment_type == 'ssl':
        train_dataset, val_dataset = build_ssl_dataset(config)
        # train_dataset, val_dataset = build_ssl_dataset_old(config)
    elif experiment_type == 'classification':
        train_dataset, val_dataset = build_classification_dataset(config)

    return train_dataset, val_dataset

def read_ssl_config(config):
    """
    Reads SSL-specific configuration parameters from the provided config object.
    Returns a dictionary of all relevant SSL configuration parameters.
    """
    cfg = config.data
    aug = cfg.augmentation
    return {
        'subset_percentage': cfg.get('subset_percentage', 1.0),
        'val_split_ratio': cfg.get('val_split_ratio', 0.1),
        'validation_path': cfg.get('validation_path', None),
        'hdf5_path': cfg.get('dataset_path', None),
        'img_reader_type': cfg.get('img_reader_type', 'legacy'),
        'shuffle_split': cfg.get('shuffle_split', False),
        'image_key': cfg.get('image_column_name', 'images'),
        'fluorescence_key': cfg.get('fluorescence_spectra_column_name', 'relative_spectra'),
        'fluo_aug_prob': aug.get('fluo_aug_prob', 0.0),
        'fl_aug_type': aug.get('fluo_augment_style', 'covariance_style_augmentation'),
        'fluo_pca_std': aug.get('fluo_pca_std', 0.1),
        'img_aug_prob': aug.get('img_aug_prob', 0.25),
        'image_transforms': aug.get('image_transforms', None),
        'legacy_augmentation': aug.get('legacy_img_aug', True),
        'image_size': cfg.get('image_size', 200),
        'image_normalization': cfg.get('image_normalization', None),
        'stitch_images': cfg.get('stitch_images', False),
        'seed': config.get('seed', 42)
    }

def image_reader_kwargs(cfg) -> dict:
    """Extract image-reader options from a ``.get``-able data-config dict.

    Wires the previously-unwired config keys ``image_size``,
    ``image_normalization`` and ``stitch_images`` into the image readers.
    Only non-default values are returned so callers can conditionally pass
    them through (a dataset may reject an option it does not support, e.g.
    ``stitch_images`` for two-image classification data).
    """
    def _scalar(value):
        if value is None:
            return None
        try:
            return float(value[0])
        except (IndexError, TypeError):
            return float(value)

    out = {'image_size': int(cfg.get('image_size', 200) or 200)}
    norm = cfg.get('image_normalization') or {}
    if norm.get('enable'):
        out['img_mean'] = _scalar(norm.get('img_mean'))
        out['img_std'] = _scalar(norm.get('img_std'))
    if cfg.get('stitch_images', False):
        out['stitch_images'] = True
    return out


def build_ssl_dataset(config):
    """
    Builds SSL dataset with flexible splitting strategies.
    Maps indices to:
       - train_dataset (WITH augmentation)
       - val_dataset (WITHOUT augmentation)
    """
    c = read_ssl_config(config)

    # Base kwargs required for both Train and Val datasets
    base_kwargs = {
        'hdf5_path': c['hdf5_path'],
        'image_key': c['image_key'],
        'fluorescence_key': c['fluorescence_key'],
        'img_reader_type': c['img_reader_type'],
        'unlabeled_stats': unlabeled_data_stats,
        **image_reader_kwargs(c),
    }

    # Instantiate lightweight version just to get total length
    total_len = len(Stage1Dataset(**base_kwargs))

    # A. Generate Base Indices
    if c['shuffle_split']:
        generator = torch.Generator().manual_seed(c['seed'])
        all_indices = torch.randperm(total_len, generator=generator).tolist()
        split_type_str = "Randomized"
    else:
        all_indices = list(range(total_len))
        split_type_str = "Sequential (Last % for Val)"

    # B. Apply Subsetting
    subset_len = int(total_len * c['subset_percentage'])
    active_indices = all_indices[:subset_len]

    print(f"Dataset Total: {total_len} | Mode: {split_type_str}")
    print(f"Using Subset ({c['subset_percentage']*100}%): {len(active_indices)} samples")

    # C. Split Train vs Validation Indices
    if c['val_split_ratio'] > 0 and c['validation_path'] is None:
        val_len = int(len(active_indices) * c['val_split_ratio'])
        train_len = len(active_indices) - val_len
        train_indices = active_indices[:train_len]
        val_indices = active_indices[train_len:]
    else:
        train_indices = active_indices
        val_indices = []

    # --- Dataset Object Creation Phase ---
    # Pass a deterministic seed to the augmentation so that image augmentations
    # are reproducible across runs. The seed is derived from the global seed and
    # an offset for the augmentation-specific RNG.
    train_aug_seed = c['seed'] + 1000
    train_source_dataset = Stage1Dataset(
        **base_kwargs,
        fluo_aug_prob=c['fluo_aug_prob'],
        img_aug_prob=c['img_aug_prob'],
        fl_aug_type=c['fl_aug_type'],
        fluo_pca_std=c['fluo_pca_std'],
        image_transforms=c['image_transforms'],
        legacy_augmentation=c['legacy_augmentation'],
        seed=train_aug_seed
    )

    val_source_dataset = Stage1Dataset(
        **base_kwargs, 
        fluo_aug_prob=0.0, 
        img_aug_prob=0.0
    )

    # --- Final Mapping ---
    train_dataset = Subset(train_source_dataset, train_indices)

    if c['validation_path'] is None and c['val_split_ratio'] > 0:
        val_dataset = Subset(val_source_dataset, val_indices)
    elif c['validation_path'] is not None:
        val_kwargs = {**base_kwargs, 'hdf5_path': c['validation_path']}
        val_dataset = Stage1Dataset(**val_kwargs, fluo_aug_prob=0.0, img_aug_prob=0.0)
    else:
        # No validation file and val_split_ratio == 0 -> no validation set
        # (the documented way to disable validation entirely).
        val_dataset = None

    print(f"✓ Train Size: {len(train_dataset)} samples (Augmented)")
    if val_dataset is not None:
        print(f"✓ Val Size:   {len(val_dataset)} samples (Clean)")
    else:
        print("✓ Val Size:   none (validation disabled)")

    return train_dataset, val_dataset



# Resolve a (possibly relative) category-map path to an existing absolute path so
# class names are found regardless of the process working directory (CWD). The
# default config ships cat_map_path relative to the project root, so a naive
# Path(cat_map_path).exists() check fails when training is launched from elsewhere
# and the confusion-matrix plot silently falls back to 'Class_0, Class_1, ...'.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _resolve_category_map_path(cat_map_path, config=None):
    """Return an existing absolute path for cat_map_path (else the original value).

    Tries the path as-is (CWD-relative), relative to the project root, and relative
    to the config file's directory (both the full relative path and just the
    basename). Returns the first candidate that exists; if none exist, returns the
    original value so callers can still warn / fall back as before.
    """
    if not cat_map_path:
        return cat_map_path
    if os.path.isabs(cat_map_path):
        return cat_map_path
    candidates = [cat_map_path, os.path.join(_PROJECT_ROOT, cat_map_path)]
    config_path = getattr(config, 'config_path', None) if config is not None else None
    if config_path:
        config_dir = os.path.dirname(config_path)
        candidates.append(os.path.join(config_dir, cat_map_path))
        candidates.append(os.path.join(config_dir, os.path.basename(cat_map_path)))
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return cat_map_path

def category_map_labels_outside_num_classes(config):
    """Pure check: which category-map labels are >= the configured ``num_classes``.

    The configured ``num_classes`` is AUTHORITATIVE: this function only
    reports which labels of the category map fall outside ``[0, num_classes)``
    (their samples are excluded from the datasets by the label filter). It
    never modifies the config and never emits any output.

    Returns:
        Sorted list of the out-of-range map labels, ``[]`` when the map is
        fully covered, or ``None`` when the check does not apply (no
        ``num_classes`` or ``cat_map_path`` in the config, or the map file
        is unreadable).
    """
    try:
        num_classes = int(config.architecture_setup.classification_model.num_classes)
    except Exception:
        return None  # num_classes not set in this config; nothing to check
    data_cfg = getattr(config, 'data', None)
    map_path = data_cfg.get('cat_map_path', None) if data_cfg is not None else None
    if not map_path:
        return None
    resolved = _resolve_category_map_path(map_path, config)
    if not resolved or not os.path.exists(resolved):
        return None
    outside = []
    try:
        with open(resolved) as fh:
            for ln in fh.read().splitlines()[1:]:
                parts = ln.strip().split('\t')
                if len(parts) == 2:
                    try:
                        label = int(parts[1])
                    except ValueError:
                        continue
                    if label >= num_classes and label not in outside:
                        outside.append(label)
    except OSError:
        return None  # unreadable map; nothing to report
    return sorted(outside)


def warn_category_map_num_classes(config, logger=None):
    """Warn ONCE when the category map holds labels >= the configured ``num_classes``.

    The configured ``num_classes`` stays authoritative: samples with
    out-of-range labels are excluded from the datasets (and their sample
    counts are reported by ``check_label_coverage``). This function emits
    the mismatch through a single channel pair: one console line via
    ``print`` and one log-file entry via ``logger`` (the ``model_logger``
    has no console handler, so the user sees exactly one line). Call it
    once per run, on rank 0 only.

    Args:
        config (EasyDict): Configuration object.
        logger: Optional logger with ``.warning()`` (the rank-0
            ``model_logger`` writes it to ``training.log``).

    Returns:
        The list of out-of-range labels when a warning was emitted, else ``None``.
    """
    outside = category_map_labels_outside_num_classes(config)
    if not outside:
        return None
    num_classes = int(config.architecture_setup.classification_model.num_classes)
    map_path = config.data.get('cat_map_path')
    msg = (
        f"num_classes={num_classes} but the category map ({map_path}) contains "
        f"label(s) {outside}; the configured num_classes is kept and samples "
        f"with those labels are excluded from the datasets (classifier head: "
        f"{num_classes} outputs). To train on them, raise "
        f"architecture_setup.classification_model.num_classes."
    )
    print(msg)
    if logger is not None:
        logger.warning(msg)
    return outside


def build_classification_dataset(config):
    """
    Builds a dataset instance for classification based on the provided configuration.

    Args:
        config (EasyDict): Configuration object.

    Returns:
        tuple: (train_dataset, val_dataset) for classification training.
    """
    cfg = config.data
    aug = cfg.augmentation

    # Pass the head's class count (the configured num_classes is
    # AUTHORITATIVE) to the datasets so samples with labels outside
    # [0, num_classes) are excluded up front (deterministic on every rank)
    # instead of crashing inside the loss; the training worker warns once
    # about a category-map mismatch (warn_category_map_num_classes, rank 0)
    # and reports the exclusions via check_label_coverage().
    try:
        head_num_classes = int(config.architecture_setup.classification_model.num_classes)
    except Exception:
        head_num_classes = None

    base_kwargs = {
        'fluorescence_key': cfg.get('fluorescence_spectra_column_name', 'relative_spectra'),
        'label_key': cfg.get('category_number_column_name', 'category_num'),
        'img_reader_type': cfg.get('img_reader_type', 'legacy'),
        'legacy_augmentation': aug.get('legacy_img_aug', True),
        'fl_aug_type': aug.get('fluo_augment_style', 'covariance_style_augmentation'),
        'fluo_pca_std': aug.get('fluo_pca_std', 0.1),
        'cat_map_path': _resolve_category_map_path(cfg.get('cat_map_path', None), config),
        **image_reader_kwargs(cfg),
    }

    train_dataset = Stage2Dataset(
        **base_kwargs,
        hdf5_path=cfg.get('train_data_path', None),
        image_key=cfg.get('image_column_name', 'images'),
        img_aug_prob=aug.get('img_aug_prob', 0.5),
        fluo_aug_prob=aug.get('fluo_aug_prob', 0.5),
        image_transforms=aug.get('image_transforms', None),
        unlabeled_stats=unlabeled_data_stats,
        num_classes=head_num_classes,
        seed=config.get('seed', 42) + 2000  # Different offset from SSL augmentation
    )

    val_dataset = Stage2Dataset(
        **base_kwargs,
        hdf5_path=cfg.get('validation_path', None),
        image_key=cfg.get('image_column_name', 'holo_image_paths'),
        img_aug_prob=0.0,
        fluo_aug_prob=0.0,
        unlabeled_stats=None,
        num_classes=head_num_classes
    )

    return train_dataset, val_dataset


def build_ssl_dataset_old(config):
    print("WARNING: Using OLD SSL dataset building logic. This is less efficient and may have data leakage between train/val if 'shuffle_split' is False. Consider updating to the new logic for better performance and cleaner splits.")
    # --- 1. Configuration ---
    subset_percentage = config.data.get('subset_percentage', 1.0)
    val_split_ratio = config.data.get('val_split_ratio', 0.1)
    validation_path = config.data.get('validation_path', None)
    data_path = config.data.get('dataset_path', None)
    img_reader_type = config.data.get('img_reader_type', 'legacy')

    shuffle_split = config.data.get('shuffle_split', False)

    image_key = config.data.get('image_column_name', 'images')
    fl_key = config.data.get('fluorescence_spectra_column_name', 'relative_spectra')

    fluo_aug_prob = config.data.augmentation.get('fluo_aug_prob', 0.0)
    fluo_augment_style = config.data.augmentation.get('fluo_augment_style', 'covariance_style_augmentation')
    img_aug_prob = config.data.augmentation.get('img_aug_prob', 0.5)
    legacy_augmentation = config.data.augmentation.get('legacy_img_aug', True)
    # seed = config.get('seed', 42)

    # Create the full dataset first
    full_dataset = Stage1Dataset(
        hdf5_path=data_path,
        image_key=image_key,
        fluorescence_key=fl_key,
        fluo_aug_prob=fluo_aug_prob,
        img_aug_prob=img_aug_prob,     # <--- ON
        legacy_augmentation=legacy_augmentation,
        image_transforms=config.data.augmentation.get('image_transforms', None),
        fl_aug_type=fluo_augment_style,
        img_reader_type=img_reader_type,
        unlabeled_stats=unlabeled_data_stats
    )

    # Apply subsetting if subset_percentage is less than 1.0
    if subset_percentage < 1.0:
        total_len = len(full_dataset)
        subset_len = int(total_len * subset_percentage)
        remaining_len = total_len - subset_len
        
        generator = torch.Generator().manual_seed(config.seed)
        full_dataset, _ = random_split(full_dataset, [subset_len, remaining_len], generator=generator)
        print(f"Using {subset_percentage*100}% of the dataset for SSL training: {len(full_dataset)} samples.")

    if val_split_ratio > 0 or validation_path is None:
        total_len = len(full_dataset)
        val_len = int(total_len * val_split_ratio)
        train_len = total_len - val_len
        
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)
        
        return train_dataset, val_dataset
    else:
        train_dataset = full_dataset # If no validation split, use the full (potentially subsetted) dataset as train
        
        val_dataset = Stage1Dataset(
        hdf5_path=data_path,
        image_key=image_key,
        fluorescence_key=fl_key,
        fluo_aug_prob=0.0,
        img_aug_prob=0.0,              # <--- OFF
        img_reader_type=img_reader_type,
        unlabeled_stats=unlabeled_data_stats
    )
        return train_dataset, val_dataset