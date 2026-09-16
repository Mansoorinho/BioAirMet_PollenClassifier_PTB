'''
@file    :   augmentation.py
@author  :   Mansoor Nabawi
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    The two augmentation pipelines used by the BioAirMet data loaders.

    1. Image augmentation
    -------------------
    AddGaussianNoise
        Standalone Gaussian-noise transform (PIL image or tensor in [0, 1]).

    ImageAugmentation_Generic
        The image augmentation pipeline (single image or image pair) with two
        modes, selected by the ``legacy`` flag:

        * ``legacy=True`` (default, config ``legacy_img_aug: True``): a fixed,
          hard-coded per-stage pipeline (``stage`` = ``'ssl'`` |
          ``'classification'``); it is not customizable and the per-transform entries
          of the ``image_transforms`` config block are ignored (the block's
          ``fill`` setting still applies, see below).
        * ``legacy=False``: built directly from the
          ``data.augmentation.image_transforms`` config block.  Each entry is a
          dict with ``enabled``/``prob`` plus parameters:
            horizontal_flip / vertical_flip  { prob }
            rotation      { prob, angle, translation: [dx, dy],
                            interpolation: 'bilinear' | 'nearest' (optional),
                            translate_round: bool (optional) }
            affine        { prob, translation: [dx, dy], scale: [lo, hi],
                            interpolation: 'bilinear' | 'nearest' (optional),
                            translate_round: bool (optional) }
            color_jitter  { prob, brightness, contrast, saturation, hue }
            gaussian_blur { prob, kernel_size, sigma: number | [lo, hi] }
            gaussian_noise{ prob, std }
            speckle_noise { prob, sigma }
            gamma_correction { prob, gamma: number | [lo, hi] }
            cutout        { prob, size: number | [lo, hi], clear_center,
                            count: number | [lo, hi] }
            Every transform is OFF by default: it only runs when its entry has
            an explicit ``enabled: true`` (an entry without ``enabled`` stays
            disabled and is reported as a warning).  This in particular holds
            for the newer speckle_noise / gamma_correction / cutout entries.
          A missing/empty block means identity (no augmentation).  ``prob`` is
          the probability of the transform ITSELF; the reader additionally gates
          the whole pipeline with ``data.img_aug_prob``, so the effective
          probability is ``img_aug_prob * prob`` (shown by ``summary()``).
          The block is validated strictly: an unknown transform name raises,
          while a missing ``enabled`` key or an unknown parameter inside an entry
          is reported as a warning (and stays disabled/ignored).
          ``parallel_flipping_rotation: true`` applies ONE shared random
          decision for the geometry (flips/rotation/affine) to both views of an
          image pair; when false each view gets its OWN geometry.  Pixel
          operations (jitter/blur/noise/speckle/gamma/cutout) are always
          independent per view.
          ``fill`` (``'white'`` | ``'black'`` | ``'border'`` | number in
          [0, 1]) controls the pixels exposed by rotation/translation in BOTH
          modes (custom and legacy) AND the pixels erased by the cutout (custom
          mode only).  When the key is absent the default is white, the
          background of this corpus.  The shipped templates use ``'border'``
          instead: the median colour of each image's outer 1-pixel ring,
          computed per image at call time (a few microseconds), so the exposed
          edge picks up the image's own edge colour.  All fill values are
          resolved per image domain ([0, 1] for float tensors, 0-255 for uint8
          PIL images).
          The corpus is 200x200 grayscale holograms with the particle centered,
          so the cutout erases only small rectangles in a ring away from the
          center (``clear_center`` keeps the middle clear), and the speckle/gamma
          ops are pixel-wise, so the particle is never removed, only textured.

        The active set is exposed via ``active_transforms`` / ``summary()`` and
        rendered for the training log by ``format_log()`` / ``log_to()``.

    _Transform / _FlipOp / _AffineOp / _PixelOp / _SpeckleOp / _GammaOp /
    _CutoutOp
        The small transform objects a custom-mode pipeline is built from
        (the legacy classification pipeline also reuses ``_AffineOp`` for its
        rotation/affine stages, with ``p=1.0``, so the same ``fill``
        resolution applies to both modes): each owns its probability gate and
        works on PIL images and on float tensors.  The geometry ops
        (flips/affine) additionally implement ``apply_pair`` so an image pair
        can share ONE random geometry decision.

    2. Fluorescence augmentation
    ---------------------------
    FluorescenceAugmentation
        A config-driven pipeline for the raw 15-dim fluorescence spectrum with a
        single probability gate.  Strategies: ``'pca_jitter'`` (PCA-space Gaussian
        jitter) or ``'covariance_style_augmentation'`` (blend toward a random
        reference style); pass ``style=None`` to disable it.  See
        :class:`FluorescenceAugmentation` and :func:`build_fluorescence_augmentation`.
    ]
'''

from __future__ import annotations

import math
import random
from typing import Any, Union, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image  # only used to resolve/report the fill for the PIL path

from .mean_std_cov_data import SSL_EIG_VALS, SSL_EIG_VECS, unlabeled_data_stats


def _entry(value: Any) -> Dict[str, Any]:
    """Normalise one ``image_transforms`` entry: bare bool -> ``{enabled: bool}``."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"enabled": bool(value)}


def _num(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AddGaussianNoise:
    """Adds Gaussian noise to an image.

    Works with either PIL images or tensors.  Noise is always applied on float
    tensors in [0, 1] and clamped back into range.
    """

    def __init__(self, std=0.04, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, img):
        if torch.rand(()) >= self.p:
            return img

        input_is_pil = not isinstance(img, torch.Tensor)
        if input_is_pil:
            img = TF.to_tensor(img)

        img = img.float()
        img = (img + torch.randn_like(img) * self.std).clamp(0.0, 1.0)

        if input_is_pil:
            img = TF.to_pil_image(img)

        return img


# -----------------------------
# Custom-mode transform objects
# -----------------------------
# A custom-mode pipeline is a plain ordered list of the objects below (see
# ``ImageAugmentation_Generic``).  Each one owns its probability gate and works
# on BOTH 8-bit PIL images (legacy reader) and float tensors in [0, 1]
# (minmax/global readers):
#
# * ``__call__`` gates and applies the transform to ONE image.
# * ``apply_pair`` applies it to the two views of a pair; pixel ops draw
#   independently per view, while the geometry ops (flips/affine) draw ONE
#   shared decision so the pair stays aligned
#   (``parallel_flipping_rotation: true``).
# * The fill for pixels exposed by rotation/translation (or erased by the
#   cutout) is resolved per image at call time via ``resolve_fill_spec``; no
#   pipeline state is mutated between calls.

def _image_size(img) -> Tuple[int, int]:
    """Return ``(width, height)`` for a PIL image or a (…, H, W) tensor."""
    if isinstance(img, torch.Tensor):
        return int(img.shape[-1]), int(img.shape[-2])
    return int(img.size[0]), int(img.size[1])  # PIL gives (width, height)


def resolve_fill_spec(fill: Any, img) -> Union[int, float]:
    """Map a fill spec to the domain of ``img``.

    ``None`` (the code default) and ``'white'`` mean the white background of
    this corpus: ``1.0`` for a float tensor in [0, 1] and ``255`` for an 8-bit
    PIL image.  A plain number is interpreted in the [0, 1] tensor domain and
    scaled to 0-255 for PIL, so both paths agree.

    ``'border'`` is content-dependent: the median of the outer 1-pixel ring
    of ``img``, computed fresh for each image at call time (a few
    microseconds), so the exposed edge picks up the colour the image itself
    has at its edge.  The result is in the same domain as the constant specs
    ([0, 1] for float tensors, 0-255 for uint8 PIL images).
    """
    is_tensor = isinstance(img, torch.Tensor)
    if fill is None:
        return 1.0 if is_tensor else 255
    if isinstance(fill, str):
        key = fill.strip().lower()
        if key == "border":
            if is_tensor:
                ring = torch.cat([
                    img[..., 0, :], img[..., -1, :],
                    img[..., :, 0], img[..., :, -1],
                ]).flatten().float()
                # median of the ring (explicit: torch.median's return type
                # varies across versions)
                s = torch.sort(ring)[0]
                n = s.numel()
                half = n // 2
                med = s[half] if n % 2 == 1 else (s[half - 1] + s[half]) * 0.5
                return med.item()
            arr = np.asarray(img)
            ring = np.concatenate([
                arr[0, :], arr[-1, :],
                arr[:, 0], arr[:, -1],
            ])
            return int(np.median(ring))
        named = {"white": (1.0, 255), "black": (0.0, 0)}
        if key not in named:
            raise ValueError(
                f"Unknown fill {fill!r}; expected 'white', 'black', 'border' or a number."
            )
        return named[key][0 if is_tensor else 1]
    value = float(fill)
    return value if is_tensor else int(round(value * 255))


class _Transform:
    """Base for custom-mode transforms: a probability gate + single-image op."""

    name = "transform"
    shared_geometry = False

    def __init__(self, p: float, meta: Optional[Dict[str, Any]] = None):
        self.p = p
        self.meta = meta if meta is not None else {"prob": p}

    def __call__(self, img):
        if torch.rand(()) >= self.p:
            return img
        return self.apply(img)

    def apply(self, img):
        raise NotImplementedError

    def apply_pair(self, a, b):
        """Default: one independent draw per view."""
        return self(a), self(b)


class _FlipOp(_Transform):
    """Horizontal or vertical flip; shares ONE decision across a pair."""

    shared_geometry = True

    def __init__(self, name: str, p: float, rng: random.Random):
        super().__init__(p)
        self.name = name
        self.rng = rng
        # TF.hflip/vflip handle PIL images and tensors; torch.flip only tensors.
        self._flip = TF.hflip if name == "horizontal_flip" else TF.vflip

    def apply(self, img):
        return self._flip(img)

    def apply_pair(self, a, b):
        if self.rng.random() < self.p:
            return self._flip(a), self._flip(b)
        return a, b


class _AffineOp(_Transform):
    """Random affine (rotation and/or translation/scale), fill per call.

    ``degrees=0`` covers the pure translation/scale ``affine`` entry; a
    non-zero ``degrees`` covers the ``rotation`` entry.  Translations are
    fractions of the image size applied in true x/y pixel directions (x scales
    with the width, y with the height).  The fill for the exposed pixels is
    resolved per image at call time (white when no ``fill`` is configured,
    border median for ``fill: 'border'``).

    ``interpolation`` (default BILINEAR) and ``translate_round`` (default
    False) let a config entry reproduce torchvision's ``RandomAffine``
    defaults (NEAREST interpolation, integer-pixel translations).  The
    hard-coded legacy pipelines pass both explicitly to match the
    pre-refactor torchvision behaviour; custom-mode entries keep the
    BILINEAR/sub-pixel defaults unless they opt in.

    ``match_rotate`` (default False) negates the angle before ``TF.affine`` so
    that a pure rotation matches ``TF.rotate``/``RandomRotation``, whose
    direction convention is the opposite of ``TF.affine``'s.
    """

    shared_geometry = True

    def __init__(self, name: str, degrees: float, translate: Tuple[float, float],
                 scale_range: Optional[Tuple[float, float]], p: float,
                 rng: random.Random, fill_spec: Any,
                 meta: Optional[Dict[str, Any]] = None,
                 interpolation: Optional[transforms.InterpolationMode] = None,
                 translate_round: bool = False,
                 match_rotate: bool = False):
        super().__init__(p, meta)
        self.name = name
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.rng = rng
        self.fill_spec = fill_spec
        # BILINEAR / sub-pixel translation are the custom-mode defaults; the
        # legacy pipelines override both to match pre-refactor torchvision.
        self.interpolation = (
            interpolation if interpolation is not None
            else transforms.InterpolationMode.BILINEAR
        )
        self.translate_round = bool(translate_round)
        # TF.rotate and TF.affine rotate in opposite directions for the same
        # angle; RandomRotation (old legacy pipeline) used TF.rotate, so the
        # angle is negated before TF.affine when this is set.
        self.match_rotate = bool(match_rotate)

    def _no_geometry(self) -> bool:
        return (self.degrees == 0 and self.scale_range is None
                and all(t == 0.0 for t in self.translate))

    def _draw(self):
        angle = self.rng.uniform(-self.degrees, self.degrees) if self.degrees else 0.0
        tx = self.rng.uniform(-self.translate[0], self.translate[0])
        ty = self.rng.uniform(-self.translate[1], self.translate[1])
        scale = self.rng.uniform(*self.scale_range) if self.scale_range else 1.0
        return angle, tx, ty, scale

    def _apply_with(self, img, angle, tx, ty, scale):
        width, height = _image_size(img)
        if self.translate_round:
            # torchvision's RandomAffine rounds translations to integer pixels.
            tx, ty = int(round(tx * width)), int(round(ty * height))
        else:
            tx, ty = tx * width, ty * height
        if self.match_rotate:
            # mirror TF.rotate's direction (see __init__)
            angle = -angle
        return TF.affine(
            img, angle=angle,
            translate=[tx, ty],  # pixels: x by width, y by height
            scale=scale, shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=resolve_fill_spec(self.fill_spec, img),
        )

    def apply(self, img):
        if self._no_geometry():
            return img
        return self._apply_with(img, *self._draw())

    def apply_pair(self, a, b):
        if self._no_geometry() or self.rng.random() >= self.p:
            return a, b
        params = self._draw()
        return self._apply_with(a, *params), self._apply_with(b, *params)


class _PixelOp(_Transform):
    """Wraps a self-random pixel transform (color jitter / Gaussian blur/noise)."""

    def __init__(self, name: str, p: float, fn, meta: Dict[str, Any]):
        super().__init__(p, meta)
        self.name = name
        self.fn = fn

    def apply(self, img):
        return self.fn(img)


class _SpeckleOp(_Transform):
    """Multiplicative speckle noise: the coherent-imaging speckle model.

    ``out = img * (1 + sigma * N(0, 1))`` clamped to [0, 1].  Multiplicative
    noise modulates the fringes like real speckle while leaving the (white)
    background in distribution, and being pixel-wise, the centered particle is
    never removed, only textured.
    """

    name = "speckle_noise"

    def __init__(self, p: float, sigma: float):
        super().__init__(p, {"prob": p, "sigma": sigma})
        self.sigma = sigma

    def apply(self, img):
        input_is_pil = not isinstance(img, torch.Tensor)
        if input_is_pil:
            img = TF.to_tensor(img)
        img = img.float()
        img = (img * (1.0 + self.sigma * torch.randn_like(img))).clamp(0.0, 1.0)
        if input_is_pil:
            img = TF.to_pil_image(img)
        return img


class _GammaOp(_Transform):
    """Gamma correction ``img ** g`` with ``g`` drawn uniformly from the range.

    The white background is a fixed point (``1.0 ** g == 1.0``), so only the
    particle intensities are re-mapped (an exposure/contrast-like variation).
    """

    name = "gamma_correction"

    def __init__(self, p: float, gamma_range: Tuple[float, float]):
        super().__init__(p, {"prob": p, "gamma": gamma_range})
        self.gamma_range = gamma_range

    def apply(self, img):
        lo, hi = self.gamma_range
        g = lo if lo == hi else float(torch.empty(1).uniform_(lo, hi))
        input_is_pil = not isinstance(img, torch.Tensor)
        if input_is_pil:
            img = TF.to_tensor(img)
        img = img.float().pow(g).clamp(0.0, 1.0)
        if input_is_pil:
            img = TF.to_pil_image(img)
        return img


class _CutoutOp(_Transform):
    """Small rectangular erases placed OUTSIDE a protected central disk.

    The particle in this corpus is centered, so no pixel within
    ``clear_center * S/2`` of the center is ever erased: each erase center is
    sampled at a random angle at a radius between
    ``clear_center * S/2 + side*sqrt(2)`` and the image edge, which keeps even
    the closest CORNER of the erased rectangle outside the protected disk.
    Per draw, ``count`` rectangles are erased (``count`` is an int, or a
    [lo, hi] range drawn uniformly per draw); every rectangle has its own size
    drawn from ``size_range`` (fraction of the image width) and its own
    position, and overlapping rectangles are harmless (same fill).  The erased
    pixels are filled with the configured fill, resolved ONCE from the
    original image before any cutting, so a rectangle reaching the border
    cannot contaminate the sampled fill colour.  With the defaults (size
    0.02-0.06 of the width, clear_center 0.6) nothing within the central 60 px
    of a 200 px image is ever modified.
    """

    name = "cutout"

    def __init__(self, p: float, size_range: Tuple[float, float],
                 clear_center: float, count_range: Tuple[int, int],
                 fill_spec: Any):
        super().__init__(p, {"prob": p, "size": size_range,
                             "count": count_range, "clear_center": clear_center})
        self.size_range = size_range
        self.clear_center = clear_center
        self.count_range = count_range
        self.fill_spec = fill_spec

    def _finish(self, img, input_is_pil):
        if input_is_pil:
            return TF.to_pil_image(img)
        return img

    def apply(self, img):
        input_is_pil = not isinstance(img, torch.Tensor)
        # Resolve the fill from the ORIGINAL image first (see class docstring).
        # For PIL input the value is in 0-255, but the work happens on the
        # [0, 1] float tensor produced by to_tensor, so scale it down.
        fill = resolve_fill_spec(self.fill_spec, img)
        if input_is_pil:
            img = TF.to_tensor(img)
            fill = fill / 255.0
        img = img.float().clone()
        height, width = img.shape[-2], img.shape[-1]

        lo_c, hi_c = self.count_range
        n_cuts = lo_c if lo_c == hi_c else int(torch.randint(lo_c, hi_c + 1, (1,)))

        half = min(width, height) / 2.0
        for _ in range(n_cuts):
            lo, hi = self.size_range
            size_frac = lo if lo == hi else float(torch.empty(1).uniform_(lo, hi))
            side = int(round(size_frac * width))
            if side < 1:
                continue
            # Keep the WHOLE erased rectangle (incl. its corners) outside the
            # protected central disk: the corner farthest from the rectangle
            # center is side*sqrt(2) away, so subtracting only side/2 would let
            # a diagonal placement clip into the disk.
            r_min = self.clear_center * half + side * math.sqrt(2)
            r_max = half - side / 2.0
            if r_max < r_min:
                # The protected center is so large that no valid spot exists.
                continue

            radius = float(torch.empty(1).uniform_(r_min, r_max))
            theta = float(torch.empty(1).uniform_(0.0, 2.0 * math.pi))
            cx = width / 2.0 + radius * math.cos(theta)
            cy = height / 2.0 + radius * math.sin(theta)

            x0 = min(max(int(round(cx - side / 2.0)), 0), width - side)
            y0 = min(max(int(round(cy - side / 2.0)), 0), height - side)
            img[..., y0:y0 + side, x0:x0 + side] = fill
        return self._finish(img, input_is_pil)


class ImageAugmentation_Generic:
    """
    Config-driven image augmentation with:
      - legacy mode (SSL / classification presets), no pair augmentation
      - custom mode built from image_transforms config
      - optional per-entry ``interpolation`` ('bilinear'/'nearest') and
        ``translate_round`` for the rotation/affine entries (custom mode)
      - the ``fill`` setting (white by default) for the pixels exposed by
        rotation/translation in BOTH modes, and by the cutout in custom mode
      - optional coordinated geometric transforms for pairs (flip/rotation)
      - detailed logging (summary() / format_log() / log_to() / log())

    Args:
        image_transforms: the ``data.augmentation.image_transforms`` config
            block (dict).  Its per-transform entries are only used in custom
            mode (``legacy=False``); the ``fill`` setting and
            ``parallel_flipping_rotation`` flag are read in both (the flag is
            a no-op in legacy mode, which has no shared pair geometry).
        legacy: ``True`` (default) runs the fixed hard-coded legacy pipeline;
            the per-transform entries of the ``image_transforms`` block are
            then ignored (the block's ``fill`` setting still applies).
        stage: which legacy pipeline to run (``'ssl'`` | ``'classification'``);
            only used when ``legacy`` is True.
    """

    # The readers pass an image pair in ONE call so the pair geometry
    # (parallel_flipping_rotation) can share the randomness across views.
    supports_pairs = True

    def __init__(self, image_transforms: Optional[Dict[str, Any]] = None,
                 legacy: bool = True, stage: str = "ssl",
                 gate_prob: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Args:
            image_transforms: the ``data.augmentation.image_transforms`` block;
                its per-transform entries are only used when ``legacy`` is
                False (the ``fill`` setting applies in both modes).
            legacy: ``True`` runs the fixed hard-coded per-stage pipeline.
            stage: ``'ssl'`` | ``'classification'`` (legacy mode only).
            gate_prob: the outer ``data.img_aug_prob`` gate the dataset's image
                reader applies (it samples ``torch.rand() < gate_prob`` before
                calling this pipeline).  Only used for reporting: it makes
                :meth:`summary` show the EFFECTIVE probability of each transform
                (``gate_prob x prob``) instead of only the per-transform one.
            seed: if provided, seed the internal random.Random used for
                geometric transforms (flip/rotation). Makes augmentation
                deterministic for reproducibility.
        """
        stage = (stage or "ssl").strip().lower()
        if stage not in ("ssl", "classification"):
            raise ValueError(
                f"Unknown augmentation stage {stage!r}; "
                "expected one of ['ssl', 'classification'].")
        self.stage = stage
        self.legacy = bool(legacy)
        self.image_transformations_cfg = dict(image_transforms or {})
        self.parallel_geom_for_pairs = self.image_transformations_cfg.get(
            "parallel_flipping_rotation", False
        )
        # Fill for pixels exposed by rotation/translation (and, in custom
        # mode, by the cutout).  It applies to BOTH modes: the custom-mode
        # rotation/affine ops, the legacy rotation/affine stages and the cutout
        # all resolve it per image at call time (see ``resolve_fill_spec``).
        # None -> white, which is the background of this corpus (see
        # ``_fill_for`` for the domain rules).  Validated here, not only in
        # custom mode, so a bad value fails in legacy mode as well.
        self.fill_spec = self.image_transformations_cfg.get("fill", None)
        self._validate_fill(self.fill_spec)
        # Outer reader gate (img_aug_prob) - reporting only.
        self.gate_prob = None if gate_prob is None else _num(gate_prob, 1.0)

        # Pipelines: ``augmentation`` is the single-image pipeline (the legacy
        # Compose, or the custom-mode Compose of the ops below); ``_ops`` is the
        # ordered custom-mode list, also used for the pair path (shared geometry).
        self.augmentation = None
        self._ops: List[Any] = []
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # Logging tracking
        self._mode: Optional[str] = None
        self._active_transforms: Dict[str, Dict[str, Any]] = {}
        self._config_warnings: List[str] = []

        # Setup transforms
        if self.legacy:
            self.__legacy_augmentation()
        else:
            self.__get_custom_transformations()

    # -------------------------
    # Legacy augmentation
    # -------------------------
    def __legacy_augmentation(self):
        self._mode = "legacy"
        self._active_transforms = {}

        self.ssl_legacy_augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            ], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        # The rotation/affine stages run through the SAME ``_AffineOp`` as
        # custom mode (p=1.0, i.e. always applied, like the old
        # ``RandomRotation``/``RandomAffine`` they replace) so the configured
        # ``fill`` (not torchvision's black default) is resolved per image for
        # the exposed pixels.
        # Pre-refactor parameters, kept deliberately: RandomRotation(25) and
        # RandomAffine(translate=(0.05, 0.05)).  Old torchvision applied both
        # with nearest interpolation and integer-pixel translations, so both
        # options are set here to mirror that behaviour (match_rotate keeps
        # the old RandomRotation direction convention).
        self.classification_legacy_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            _AffineOp("rotation", degrees=25.0, translate=(0.0, 0.0),
                      scale_range=None, p=1.0, rng=self._rng,
                      fill_spec=self.fill_spec, meta={"degrees": 25},
                      interpolation=transforms.InterpolationMode.NEAREST,
                      translate_round=True, match_rotate=True),
            _AffineOp("affine", degrees=0.0, translate=(0.05, 0.05),
                      scale_range=None, p=1.0, rng=self._rng,
                      fill_spec=self.fill_spec, meta={"translate": (0.05, 0.05)},
                      interpolation=transforms.InterpolationMode.NEAREST,
                      translate_round=True),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.7, contrast=0.7)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5))
            ], p=0.5)
        ])

        if self.stage == "ssl":
            self.augmentation = self.ssl_legacy_augmentation
            self._active_transforms = {
                "color_jitter": {"prob": 0.5, "brightness": 0.5, "contrast": 0.5},
                "gaussian_blur": {"prob": 0.5, "kernel_size": 3, "sigma": (0.1, 2.0)},
                "horizontal_flip": {"prob": 0.5},
                "vertical_flip": {"prob": 0.5},
            }
        else:
            self.augmentation = self.classification_legacy_augmentation
            self._active_transforms = {
                "horizontal_flip": {"prob": 0.5},
                "vertical_flip": {"prob": 0.5},
                "rotation": {"degrees": 25},
                "affine": {"translate": (0.05, 0.05)},
                "color_jitter": {"prob": 0.8, "brightness": 0.7, "contrast": 0.7},
                "gaussian_blur": {"prob": 0.5, "kernel_size": 3, "sigma": (0.1, 2.5)},
            }

    # -------------------------
    # Custom augmentation
    # -------------------------
    # The transforms accepted under data.augmentation.image_transforms when
    # legacy_img_aug is False, in the order the pipeline runs them.
    _TRANSFORMS = (
        "horizontal_flip", "vertical_flip", "rotation", "affine",
        "color_jitter", "gaussian_blur", "gaussian_noise",
        "speckle_noise", "gamma_correction", "cutout",
    )
    _SUPPORTED_KEYS = ("parallel_flipping_rotation", "fill", *_TRANSFORMS)
    # Per-entry parameters (typos inside an entry are reported as warnings).
    _ENTRY_PARAMS = {
        "horizontal_flip": ("enabled", "prob"),
        "vertical_flip": ("enabled", "prob"),
        "rotation": ("enabled", "prob", "angle", "translation",
                     "interpolation", "translate_round"),
        "affine": ("enabled", "prob", "translation", "scale",
                   "interpolation", "translate_round"),
        "color_jitter": ("enabled", "prob", "brightness", "contrast", "saturation", "hue"),
        "gaussian_blur": ("enabled", "prob", "kernel_size", "sigma"),
        "gaussian_noise": ("enabled", "prob", "std"),
        "speckle_noise": ("enabled", "prob", "sigma"),
        "gamma_correction": ("enabled", "prob", "gamma"),
        "cutout": ("enabled", "prob", "size", "clear_center", "count"),
    }

    def _warn(self, msg: str) -> None:
        """Record (and print) a config problem that does not stop the run."""
        self._config_warnings.append(msg)
        print(f"[augmentation] WARNING: {msg}")

    def __validate_transforms(self, cfg: Dict[str, Any]) -> None:
        """Strict validation of the ``image_transforms`` block (custom mode).

        Raises ``ValueError`` for anything that would otherwise silently do the
        wrong thing (unknown transform name, non-numeric parameter, probability
        outside [0, 1]); records warnings for survivable surprises (an entry with
        no ``enabled`` key stays DISABLED - as it always was - but now says so,
        and unknown parameters inside an entry are listed).
        """
        unknown = [k for k in cfg if k not in self._SUPPORTED_KEYS]
        if unknown:
            raise ValueError(
                f"Unknown key(s) in data.augmentation.image_transforms: {sorted(unknown)}.\n"
                f"Supported keys: {sorted(self._SUPPORTED_KEYS)}.\n"
                "Transform names are exact (e.g. 'horizontal_flip', not 'hflip'). Remember that "
                "legacy mode ignores the per-transform entries (the block's fill "
                "setting still applies) - set legacy_img_aug: false to use them."
            )

        for key in self._ENTRY_PARAMS:
            if key not in cfg:
                continue
            entry = cfg[key]
            if isinstance(entry, bool) or entry is None:
                continue
            if not isinstance(entry, dict):
                raise ValueError(
                    f"image_transforms.{key} must be a bool or a mapping, got "
                    f"{type(entry).__name__}: {entry!r}"
                )
            extra = [k for k in entry if k not in self._ENTRY_PARAMS[key]]
            if extra:
                self._warn(
                    f"image_transforms.{key} has unknown parameter(s) {sorted(extra)}; "
                    f"supported: {sorted(self._ENTRY_PARAMS[key])}. They are ignored."
                )
            if "enabled" not in entry:
                self._warn(
                    f"image_transforms.{key} has no 'enabled' key -> the transform is DISABLED "
                    "(add 'enabled: true' to turn it on)."
                )
            if "prob" in entry:
                p_raw = entry["prob"]
                p = _num(p_raw, float("nan"))
                if not (p == p) or not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"image_transforms.{key}.prob must be a number in [0, 1], got {p_raw!r}"
                    )

    def _validate_fill(self, spec: Any) -> None:
        """Validate an explicit ``fill`` value (see :meth:`_fill_for`)."""
        if spec is None:
            return
        if isinstance(spec, str):
            if spec.strip().lower() not in ("white", "black", "border"):
                raise ValueError(
                    f"image_transforms.fill must be 'white', 'black', 'border' "
                    f"or a number in [0, 1]; got {spec!r}"
                )
            return
        if isinstance(spec, bool) or isinstance(spec, (list, tuple, dict)):
            raise ValueError(
                f"image_transforms.fill must be 'white', 'black', 'border' or a "
                f"number in [0, 1] (tensor domain); got {spec!r}"
            )
        v = _num(spec, float("nan"))
        if not (v == v) or not (0.0 <= v <= 1.0):
            raise ValueError(
                f"image_transforms.fill must be in [0, 1] (the [0,1] tensor domain); got {spec!r}"
            )

    def _fill_for(self, img) -> Union[int, float]:
        """Fill value for ``img``'s domain (see :func:`resolve_fill_spec`)."""
        return resolve_fill_spec(self.fill_spec, img)

    @staticmethod
    def _pair2(value: Any, name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        """Read a two-element list/tuple (or scalar) from config with validation."""
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(
                    f"image_transforms.{name} must have exactly 2 values [x, y], got {value!r}"
                )
            return _num(value[0], default[0]), _num(value[1], default[1])
        v = _num(value, default[0])
        return v, v

    def __get_custom_transformations(self):
        self._mode = "custom"
        self._active_transforms = {}

        cfg = self.image_transformations_cfg
        self.__validate_transforms(cfg)

        # One builder per transform (in pipeline order); an absent/disabled
        # entry returns None.  A missing/empty block means identity.
        self._ops = [
            op
            for name in self._TRANSFORMS
            if (op := getattr(self, f"_build_{name}")(cfg.get(name))) is not None
        ]
        self._active_transforms = {op.name: op.meta for op in self._ops}
        self.augmentation = (transforms.Compose(self._ops) if self._ops
                             else transforms.Lambda(lambda x: x))

    # -- Custom-mode builders: one per transform ---------------------------
    # Each parses + validates its config entry and returns the transform
    # object, or None when the entry is absent/disabled.

    def _build_horizontal_flip(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        return _FlipOp("horizontal_flip", _num(entry.get("prob"), 0.5), self._rng)

    def _build_vertical_flip(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        return _FlipOp("vertical_flip", _num(entry.get("prob"), 0.5), self._rng)

    def _affine_options(self, entry: Dict[str, Any], prefix: str):
        """Parse the optional ``interpolation`` / ``translate_round`` keys.

        Returns ``(interpolation, translate_round, meta_extra)``.  The defaults
        (BILINEAR, sub-pixel translations) keep the legacy pipelines and all
        existing configs bit-identical; setting ``interpolation: nearest`` and
        ``translate_round: true`` reproduces torchvision's ``RandomAffine``
        exactly (its defaults).
        """
        raw_interp = entry.get("interpolation", "bilinear")
        if not isinstance(raw_interp, str):
            raise ValueError(
                f"image_transforms.{prefix}.interpolation must be a string "
                f"('bilinear' or 'nearest'), got {raw_interp!r}"
            )
        key = raw_interp.strip().lower()
        if key == "bilinear":
            interp = transforms.InterpolationMode.BILINEAR
        elif key == "nearest":
            interp = transforms.InterpolationMode.NEAREST
        else:
            raise ValueError(
                f"image_transforms.{prefix}.interpolation must be 'bilinear' "
                f"or 'nearest', got {raw_interp!r}"
            )
        translate_round = bool(entry.get("translate_round", False))
        meta = {}
        if key != "bilinear":
            meta["interpolation"] = key
        if translate_round:
            meta["translate_round"] = True
        return interp, translate_round, meta

    def _build_rotation(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        angle = _num(entry.get("angle"), 0.0)
        tx, ty = self._pair2(entry.get("translation"), "rotation.translation", (0.0, 0.0))
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            raise ValueError(
                "image_transforms.rotation.translation must be fractions of the image "
                f"size in [0, 1], got {(tx, ty)}"
            )
        interp, translate_round, extra = self._affine_options(entry, "rotation")
        meta = {"prob": p, "angle": angle, "translation": (tx, ty), **extra}
        return _AffineOp("rotation", angle, (tx, ty), None, p, self._rng,
                         self.fill_spec, meta=meta,
                         interpolation=interp, translate_round=translate_round)

    def _build_affine(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        tx, ty = self._pair2(entry.get("translation"), "affine.translation", (0.1, 0.1))
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            raise ValueError(
                "image_transforms.affine.translation must be fractions of the image "
                f"size in [0, 1], got {(tx, ty)}"
            )
        scale_range: Optional[Tuple[float, float]] = None
        if entry.get("scale") is not None:
            lo, hi = self._pair2(entry["scale"], "affine.scale", (1.0, 1.0))
            scale_range = (min(lo, hi), max(lo, hi))
        interp, translate_round, extra = self._affine_options(entry, "affine")
        meta = {"prob": p, "translation": (tx, ty), **extra}
        if scale_range is not None:
            meta["scale"] = scale_range
        return _AffineOp("affine", 0.0, (tx, ty), scale_range, p, self._rng,
                         self.fill_spec, meta=meta,
                         interpolation=interp, translate_round=translate_round)

    def _build_color_jitter(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        b = _num(entry.get("brightness"), 0.0)
        c = _num(entry.get("contrast"), 0.0)
        s = _num(entry.get("saturation"), 0.0)
        h = _num(entry.get("hue"), 0.0)
        for nm, val, lim in (("brightness", b, None), ("contrast", c, None),
                             ("saturation", s, None), ("hue", h, 0.5)):
            if val < 0:
                raise ValueError(f"image_transforms.color_jitter.{nm} must be >= 0, got {val}")
            if lim is not None and val > lim:
                raise ValueError(
                    f"image_transforms.color_jitter.{nm} must be <= {lim} "
                    f"(torchvision limit), got {val}"
                )
        if s > 0 or h > 0:
            self._warn(
                "color_jitter.saturation/hue are NO-OPS for this single-channel "
                f"(grayscale) corpus (saturation={s}, hue={h} ignored by torchvision for "
                "1-channel images); use brightness/contrast instead."
            )
        # 0.0 is the identity factor per channel (torchvision rejects None).
        jitter = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
        return _PixelOp("color_jitter", p, jitter,
                        {"prob": p, "brightness": b, "contrast": c,
                         "saturation": s, "hue": h})

    def _build_gaussian_blur(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        ks = int(_num(entry.get("kernel_size"), 3))
        if ks < 1:
            raise ValueError(
                f"image_transforms.gaussian_blur.kernel_size must be >= 1, got {ks}"
            )
        ks = ks + 1 if ks % 2 == 0 else ks
        sigma = entry.get("sigma", [0.1, 2.0])
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError(
                    "image_transforms.gaussian_blur.sigma must be a number or a "
                    f"[min, max] pair, got {sigma!r}"
                )
            s_low, s_high = _num(sigma[0], 0.1), _num(sigma[1], 2.0)
        else:
            s_low = s_high = _num(sigma, 0.1)
        if s_low < 0 or s_high < s_low:
            raise ValueError(
                "image_transforms.gaussian_blur.sigma must satisfy 0 <= min <= max, "
                f"got {sigma!r}"
            )
        blur = transforms.GaussianBlur(kernel_size=ks, sigma=(s_low, s_high))
        return _PixelOp("gaussian_blur", p, blur,
                        {"prob": p, "kernel_size": ks, "sigma": (s_low, s_high)})

    def _build_gaussian_noise(self, raw):
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        std = _num(entry.get("std"), 0.05)
        if std < 0:
            raise ValueError(f"image_transforms.gaussian_noise.std must be >= 0, got {std}")
        # p=1.0 inside: the gate lives on the wrapper, so the noise always runs
        # when the gate opens (same object the hand-written pipelines use).
        return _PixelOp("gaussian_noise", p, AddGaussianNoise(std=std, p=1.0),
                        {"prob": p, "std": std})

    def _build_speckle_noise(self, raw):
        """OFF by default: only runs when the entry has ``enabled: true``."""
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        sigma = _num(entry.get("sigma"), 0.05)
        if sigma < 0:
            raise ValueError(f"image_transforms.speckle_noise.sigma must be >= 0, got {sigma}")
        return _SpeckleOp(p, sigma)

    def _build_gamma_correction(self, raw):
        """OFF by default: only runs when the entry has ``enabled: true``."""
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        gamma = entry.get("gamma", [0.7, 1.3])
        if isinstance(gamma, (list, tuple)):
            if len(gamma) != 2:
                raise ValueError(
                    "image_transforms.gamma_correction.gamma must be a number or a "
                    f"[min, max] pair, got {gamma!r}"
                )
            lo, hi = _num(gamma[0], 0.7), _num(gamma[1], 1.3)
        else:
            lo = hi = _num(gamma, 1.0)
        if lo <= 0 or hi < lo:
            raise ValueError(
                "image_transforms.gamma_correction.gamma must satisfy 0 < min <= max, "
                f"got {gamma!r}"
            )
        return _GammaOp(p, (lo, hi))

    def _build_cutout(self, raw):
        """OFF by default: only runs when the entry has ``enabled: true``."""
        entry = _entry(raw)
        if not entry.get("enabled", False):
            return None
        p = _num(entry.get("prob"), 0.5)
        size = entry.get("size", [0.02, 0.06])
        if isinstance(size, (list, tuple)):
            if len(size) != 2:
                raise ValueError(
                    "image_transforms.cutout.size must be a number or a "
                    f"[min, max] pair, got {size!r}"
                )
            lo, hi = _num(size[0], 0.02), _num(size[1], 0.06)
        else:
            lo = hi = _num(size, 0.02)
        if lo <= 0 or hi < lo:
            raise ValueError(
                "image_transforms.cutout.size must satisfy 0 < min <= max, "
                f"got {size!r}"
            )
        clear = _num(entry.get("clear_center"), 0.6)
        if not (0.0 <= clear < 1.0):
            raise ValueError(
                f"image_transforms.cutout.clear_center must be in [0, 1), got {clear}"
            )
        count = entry.get("count", 1)
        if count is None:  # an empty YAML key means "use the default"
            count = 1
        if isinstance(count, (list, tuple)):
            if len(count) != 2:
                raise ValueError(
                    "image_transforms.cutout.count must be a positive integer or a "
                    f"[min, max] pair, got {count!r}"
                )
            lo_c, hi_c = _num(count[0], float("nan")), _num(count[1], float("nan"))
        else:
            lo_c = hi_c = _num(count, float("nan"))
        if lo_c != lo_c or hi_c != hi_c:
            raise ValueError(
                "image_transforms.cutout.count must be a positive integer or a "
                f"[min, max] pair of integers, got {count!r}"
            )
        if lo_c != int(lo_c) or hi_c != int(hi_c):
            raise ValueError(
                f"image_transforms.cutout.count must be whole number(s), got {count!r}"
            )
        lo_c, hi_c = int(lo_c), int(hi_c)
        if lo_c < 1 or hi_c < lo_c:
            raise ValueError(
                "image_transforms.cutout.count must satisfy 1 <= min <= max, "
                f"got {count!r}"
            )
        return _CutoutOp(p, (lo, hi), clear, (lo_c, hi_c), self.fill_spec)

    # -------------------------
    # Call interface
    # -------------------------
    def __call__(
        self,
        img: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ):
        # Single image: the pipeline (legacy Compose or custom Compose of ops).
        if not (isinstance(img, (list, tuple)) and len(img) == 2):
            return self.augmentation(img)

        img1, img2 = img
        if self._mode == "legacy":
            # The fixed per-stage pipeline, applied independently per view,
            # as the old hard-coded pipelines did.
            return self.augmentation(img1), self.augmentation(img2)

        # Custom mode: same order as the single-image pipeline.  Geometry ops
        # share ONE random decision across the pair when
        # parallel_flipping_rotation is true; pixel ops (and geometry with the
        # flag off) draw independently per view, so the two views stay two
        # independent samples.
        for op in self._ops:
            if op.shared_geometry and self.parallel_geom_for_pairs:
                img1, img2 = op.apply_pair(img1, img2)
            else:
                img1, img2 = op(img1), op(img2)
        return img1, img2

    # -------------------------
    # Introspection (used by the training log)
    # -------------------------
    @property
    def active_transforms(self) -> List[str]:
        return list(self._active_transforms)

    @property
    def config_warnings(self) -> List[str]:
        """Problems found in the ``image_transforms`` block (custom mode)."""
        return list(self._config_warnings)

    def effective_prob(self, p: Optional[float]) -> Optional[float]:
        """Probability of a transform actually running, including the gate.

        The dataset's image reader applies an outer gate
        (``data.img_aug_prob``) before this pipeline is called, so the chance
        that a transform with probability ``p`` really runs is
        ``img_aug_prob * p``.  Returns ``None`` when the gate is unknown.
        """
        if p is None:
            return None
        gate = 1.0 if self.gate_prob is None else self.gate_prob
        return gate * p

    def _fill_description(self) -> str:
        """Human-readable resolved fill for both image domains."""
        if self.fill_spec is None:
            return "white (default: 1.0 for [0,1] float tensors, 255 for uint8 PIL)"
        if isinstance(self.fill_spec, str):
            key = self.fill_spec.strip().lower()
            if key == "border":
                return ("border (median of each image's outer 1-pixel ring, "
                        "computed per image at call time: [0, 1] for float "
                        "tensors, 0-255 for uint8 PIL)")
            return (f"{key} (from image_transforms.fill: "
                    f"{self._fill_for(torch.zeros(1, 2, 2)):.3g} tensor / "
                    f"{self._fill_for(Image.new('L', (2, 2)))} PIL)")
        return (f"{float(self.fill_spec):.3g} (from image_transforms.fill, [0,1] domain -> "
                f"{self._fill_for(torch.zeros(1, 2, 2)):.3g} tensor / "
                f"{self._fill_for(Image.new('L', (2, 2)))} PIL)")

    def summary(self) -> List[str]:
        """Mode line + one human-readable line per active transform.

        Reports ``none (identity)`` when nothing is active.  Probabilities are
        shown as configured AND effective (after the ``img_aug_prob`` gate), and
        the fill used for pixels exposed by rotation/translation or erased by
        the cutout is reported with its resolved value for both image domains.
        """
        lines = [f"mode: {self._mode}"]
        if self._mode == "custom":
            geometry = ("shared: one random decision for BOTH views"
                        if self.parallel_geom_for_pairs else
                        "independent: each view draws its own geometry")
            lines.append(f"parallel_flipping_rotation: {bool(self.parallel_geom_for_pairs)} "
                         f"({geometry})")
        fill_what = ("pixels exposed by rotation/translation, cutout"
                     if self._mode == "custom"
                     else "pixels exposed by rotation/translation")
        lines.append(f"fill ({fill_what}): {self._fill_description()}")
        if self.gate_prob is not None:
            lines.append(
                f"pipeline gate (data.img_aug_prob): {self.gate_prob:g}"
                + ("  -> effective probability = gate x per-transform probability"
                   if self.gate_prob < 1.0 else ""))
        if not self._active_transforms:
            lines.append("none (identity)")
        else:
            for name, details in self._active_transforms.items():
                p = details.get("prob", None)
                shown = {k: v for k, v in details.items() if k != "prob"}
                params = ", ".join(f"{k}={v}" for k, v in shown.items())
                prob_part = f"p={p}" if p is not None else ""
                eff = self.effective_prob(p)
                if eff is not None and self.gate_prob not in (None, 1.0):
                    prob_part += f", effective p={eff:.3f} (gate {self.gate_prob:g} x {p})"
                if params:
                    prob_part += f", {params}" if prob_part else params
                lines.append(f"{name}: true   ({prob_part})")
        if self._config_warnings:
            lines.append(f"config warnings ({len(self._config_warnings)}):")
            lines.extend(f"  - {w}" for w in self._config_warnings)
        return lines

    # -- training-log rendering (used by BaseTrainerV2) ------------------------
    def format_log(self, *, header: str = "--- Data Augmentation (TRAIN) ---",
                   pre_lines: Optional[List[str]] = None,
                   post_lines: Optional[List[str]] = None) -> str:
        """Build the full augmentation log block (for ``training.log``).

        ``pre_lines`` (e.g. img_aug_prob, image reader) are placed before the
        transform list and ``post_lines`` (e.g. fluorescence augmentation) after.
        """
        lines = [header]
        lines.extend(pre_lines or [])
        lines.extend("    " + line for line in self.summary())
        lines.extend(post_lines or [])
        return "\n".join(lines)

    def log_to(self, logger: Any, **kwargs: Any) -> None:
        """Log the active pipeline block via ``logger.info``.

        ``logger`` may be any object with an ``info`` method (e.g. the trainer's
        ``model_logger`` which writes to ``training.log``); ``None`` is a no-op.
        Extra keyword arguments are forwarded to :meth:`format_log`.
        """
        if logger is not None:
            logger.info(self.format_log(**kwargs))

    # -- interactive helper -----------------------------------------------------
    def log(self):
        """Print the augmentation mode and the active transform hyperparameters."""
        print(self.format_log(header="=== Augmentation Summary ==="))

    def __repr__(self) -> str:
        on = ", ".join(self.active_transforms) or "none"
        return (f"ImageAugmentation_Generic(mode={self._mode!r}, "
                f"active=[{on}])")


# ===========================================================================
# Fluorescence-spectrum augmentation
# ===========================================================================
# The fluorescence strategies operate on the raw 15-dim spectrum (the dataset
# reduces them to the model-facing 13-dim afterwards).  ``FluorescenceAugmentation``
# (below) is the single, config-driven entry point: one probability gate + one
# strategy.  ``FluorescencePCAJitter`` and ``covariance_style_augmentation`` are
# the two underlying strategies.


class FluorescencePCAJitter:
    """PCA jitter augmentor for raw 15-dim fluorescence spectra.

    I/O contract: 15-dim float tensor in -> 15-dim non-negative float tensor out.

    The optional ``probability`` gate is for standalone use only. When applied
    through :class:`FluorescenceAugmentation`, that pipeline owns the single
    probability gate and constructs the jitter with ``probability=1.0`` so it is
    applied exactly once per sampled draw.
    """

    def __init__(self, eig_vals=SSL_EIG_VALS, eig_vecs=SSL_EIG_VECS, std=0.1, probability=0.8):
        """
        eig_vals: numpy array (15,)
        eig_vecs: numpy array (15, 15)
        std: The magnitude of the jitter (0.1 is standard)
        probability: P(jitter applied) for standalone use (0.8 is standard);
            pass 1.0 when the surrounding pipeline provides the gate.
        """
        # Convert to torch buffers for speed
        self.eig_vals = torch.from_numpy(eig_vals).float()
        self.eig_vecs = torch.from_numpy(eig_vecs).float()
        self.std = std
        self.prob = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor (15,)
        Output: Tensor (15,) augmented
        """
        # Random chance to skip augmentation (keep original)
        if torch.rand(1).item() > self.prob:
            return tensor

        # Ensure float
        x = tensor.float()

        # 1. Generate random noise (alphas)
        # One random number per eigenvector (component)
        # Drawn from Normal(0, std)
        alphas = torch.normal(mean=0.0, std=self.std,
                              size=(self.eig_vals.shape[0],))

        # 2. Calculate the Jitter Vector
        # Formula: sum( alpha_i * eigenvalue_i * eigenvector_i )
        # This scales the noise by how "important" that direction is.

        # (15,) * (15,) = (15,)
        weights = alphas * self.eig_vals

        # Matrix multiplication: (15, 15) @ (15, 1) -> (15,)
        # We push the data along the principal components
        delta = torch.matmul(self.eig_vecs, weights.unsqueeze(1)).squeeze()

        # 3. Add to original image
        x_aug = x + delta

        # 4. Physics check: Fluorescence cannot be negative
        return torch.clamp(x_aug, min=0.0)


def covariance_style_augmentation(
    original_tensor: np.ndarray,
    target_styles: Dict[str, Dict[str, np.ndarray]],
    ) -> torch.Tensor:
    """
    Performs style transfer using covariance matrices for hyper-realistic augmentation.

    This function generates a new sample from a multivariate normal distribution
    defined by a target style's mean and covariance, creating noise that respects
    inter-feature correlations.
    """
    # --- Stage 1: Select a Single Target Style ---
    style_names = list(target_styles.keys())
    target_style_name = random.choice(style_names)
    target_style = target_styles[target_style_name]

    mu_target_np = target_style['mean']
    cov_target_np = target_style['cov']

    # --- Stage 2: Generate a Correlated Synthetic Sample ---
    try:
        synthetic_tensor_np = np.random.multivariate_normal(
            mean=mu_target_np, cov=cov_target_np)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: If sampling fails (non-PSD / invalid matrix, rare),
        # use the diagonal (variances) for independent noise.
        # np.maximum (not in-place masking) because the covariance array may be
        # read-only (e.g. h5py-backed views).
        stds = np.sqrt(np.maximum(np.diag(cov_target_np), 0))
        synthetic_tensor_np = np.random.normal(loc=mu_target_np, scale=stds)

    # --- Stage 3: Blend with Original Tensor (Alpha Blending) ---
    alpha = np.random.uniform(0.3, 0.8)  # Use a random alpha for more variety

    original_tensor_torch = torch.from_numpy(original_tensor).float()
    synthetic_tensor_torch = torch.from_numpy(synthetic_tensor_np).float()

    final_augmented_tensor = (
        1 - alpha) * original_tensor_torch + alpha * synthetic_tensor_torch

    return torch.clamp(final_augmented_tensor, min=0).float()


class FluorescenceAugmentation:
    """Config-driven fluorescence-spectrum augmentation (single probability gate).

    This is the single entry point used by the datasets.  It wraps the two
    supported strategies behind one object and applies **at most one** probability
    gate per draw (no double gating):

        ``style=None``                             -> identity (no augmentation)
        ``style='pca_jitter'`` (alias: ``'pca'``)  -> PCA-space Gaussian jitter
        ``style='covariance_style_augmentation'``  -> blend toward a random style

    The strategies work on the raw 15-dim spectrum; the dataset performs the
    15 -> 13 reduction separately.

    Args:
        style: augmentation strategy (``None`` to disable).
        prob:  probability of applying the augmentation (the single gate).
        std:   std-dev of the Gaussian jitter (``pca_jitter`` only).
        styles: reference style statistics ``{name: {'mean': ..., 'cov': ...}}``
            for ``covariance_style_augmentation``; if ``None`` it falls back to
            the packaged :data:`unlabeled_data_stats`.
    """

    STYLES = ('pca_jitter', 'covariance_style_augmentation')
    _ALIASES = {'pca': 'pca_jitter'}

    def __init__(self, style=None, prob: float = 0.5, std: float = 0.1, styles=None):
        self.prob = float(prob)
        self.std = float(std)
        self._styles = styles
        self.set_style(style)

    def set_style(self, style) -> None:
        """Validate and set the strategy (rebuilding the PCA jitter if needed)."""
        style = self._ALIASES.get(style, style) if style else style
        if style not in (None, *self.STYLES):
            raise ValueError(
                f"Unknown fl_aug_type: {style!r}. "
                "Supported: None, 'pca_jitter' (alias: 'pca'), "
                "'covariance_style_augmentation'.")
        self.style = style
        # The jitter is built with probability=1.0: the single gate lives here.
        self._jitter = (FluorescencePCAJitter(probability=1.0, std=self.std)
                        if style == 'pca_jitter' else None)

    @property
    def pca_jitter(self):
        """The underlying PCA jitter augmentor (``None`` unless style is ``pca_jitter``)."""
        return self._jitter

    def __call__(self, spectra_values):
        """Apply the gate + strategy to a raw 15-dim spectrum (array or tensor)."""
        if self.style is None:
            return spectra_values
        # Single probability gate.
        if torch.rand(1).item() >= self.prob:
            return spectra_values
        if self.style == 'pca_jitter':
            return self._jitter(torch.as_tensor(spectra_values, dtype=torch.float32))
        styles = self._styles if self._styles is not None else unlabeled_data_stats
        return covariance_style_augmentation(
            np.asarray(spectra_values, dtype=np.float32), target_styles=styles)

    def summary(self) -> str:
        """One-line description (for the training log)."""
        if self.style is None:
            return "fluorescence augmentation: disabled"
        if self.style == 'pca_jitter':
            return f"fluorescence augmentation: pca_jitter (std={self.std}, p={self.prob})"
        return f"fluorescence augmentation: covariance_style_augmentation (p={self.prob})"

    def __repr__(self) -> str:
        return (f"FluorescenceAugmentation(style={self.style!r}, prob={self.prob}, "
                f"std={self.std})")


def build_fluorescence_augmentation(style=None, prob: float = 0.5, std: float = 0.1,
                                    styles=None) -> "FluorescenceAugmentation":
    """Single construction point for the fluorescence augmentation pipeline.

    Args:
        style:  ``'pca_jitter'`` | ``'covariance_style_augmentation'`` | ``None``.
        prob:   single probability gate for the augmentation.
        std:    jitter std-dev (``pca_jitter``).
        styles: reference style statistics for ``covariance_style_augmentation``.
    """
    return FluorescenceAugmentation(style=style, prob=prob, std=std, styles=styles)
