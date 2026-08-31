"""
Dataset utilities for CVI-COD.

Provides CODDataset for loading triplets (image, template, gt) from directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Default configuration
INPUT_SIZE = 416
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CODDataset(Dataset):
    """
    Dataset for Camouflaged Object Detection.

    Loads pairs (original image, template) plus optional ground-truth masks
    from parallel directories. Files are matched by their path relative to
    each directory, ignoring the extension. When gt_dir is given, only samples
    with a matching ground-truth mask are kept.

    Args:
        image_dir: Directory containing original images
        template_dir: Directory containing generated templates (mental images)
        gt_dir: Directory containing ground truth masks (None for inference-only)
        input_size: Target size for resizing (default: 416)
        enable_augment: Whether to enable geometric augmentation (default: False)

    Directory structure example:
        image_dir/
            sample1.jpg
            sample2.png
        template_dir/
            sample1.png
            sample2.png
        gt_dir/
            sample1.png
            sample2.png
    """

    def __init__(
        self,
        image_dir: Path,
        template_dir: Path,
        gt_dir: Optional[Path] = None,
        input_size: int = INPUT_SIZE,
        enable_augment: bool = False,
    ) -> None:
        image_dir = Path(image_dir)
        template_dir = Path(template_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        image_map = self._scan_dir(image_dir)
        template_map = self._scan_dir(template_dir)

        gt_map: Dict[str, Path] = {}
        if gt_dir is not None:
            gt_dir = Path(gt_dir)
            if not gt_dir.exists():
                raise FileNotFoundError(f"GT directory not found: {gt_dir}")
            gt_map = self._scan_dir(gt_dir)

        common_keys = sorted(set(image_map.keys()) & set(template_map.keys()))
        if gt_dir is not None:
            common_keys = [k for k in common_keys if k in gt_map]

        if not common_keys:
            raise ValueError(
                f"No matching samples found. "
                f"Images: {len(image_map)}, Templates: {len(template_map)}, GTs: {len(gt_map)}"
            )

        self.records: List[Dict[str, Any]] = []
        for key in common_keys:
            self.records.append({
                "image_path": str(image_map[key]),
                "template_path": str(template_map[key]),
                "gt_path": str(gt_map[key]) if key in gt_map else "",
            })

        self.enable_augment = enable_augment
        self.input_size = input_size
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def _scan_dir(self, directory: Path) -> Dict[str, Path]:
        """Recursively scan for images, keyed by extension-less relative path."""
        mapping: Dict[str, Path] = {}
        for f in directory.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                mapping[f.relative_to(directory).with_suffix("").as_posix()] = f
        return mapping

    def __len__(self) -> int:
        return len(self.records)

    def _resize_triplet(
        self,
        img: np.ndarray,
        template: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Resize all inputs to target size."""
        target_size = (self.input_size, self.input_size)
        img_resized = np.array(Image.fromarray(img).resize(target_size, Image.BILINEAR))
        template_resized = np.array(Image.fromarray(template).resize(target_size, Image.BILINEAR))
        mask_resized = None
        if mask is not None:
            mask_resized = np.array(Image.fromarray(mask).resize(target_size, Image.NEAREST))
        return img_resized, template_resized, mask_resized

    def _augment_sample(
        self,
        img: np.ndarray,
        template: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply synchronized geometric augmentation."""
        hflip = np.random.rand() < 0.5
        vflip = np.random.rand() < 0.2
        rot_k = np.random.randint(0, 4) if np.random.rand() < 0.25 else 0

        def _apply(x: np.ndarray) -> np.ndarray:
            out = x
            if hflip:
                out = np.flip(out, axis=1)
            if vflip:
                out = np.flip(out, axis=0)
            if rot_k > 0:
                out = np.rot90(out, k=rot_k, axes=(0, 1))
            return np.ascontiguousarray(out)

        img_aug = _apply(img)
        template_aug = _apply(template)
        mask_aug = _apply(mask) if mask is not None else None
        return img_aug, template_aug, mask_aug

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """Convert to tensor with ImageNet normalization."""
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _mask_to_tensor(self, mask_np: np.ndarray) -> torch.Tensor:
        """Convert binary mask to tensor [1, H, W]."""
        return torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        # Load images
        with Image.open(record["image_path"]) as img:
            img_np = np.array(img.convert("RGB"))
        with Image.open(record["template_path"]) as img:
            template_np = np.array(img.convert("RGB"))

        mask_np = None
        gt_path = record["gt_path"]
        if gt_path:
            with Image.open(gt_path) as img:
                mask_np = np.array(img.convert("L"))

        orig_size = (img_np.shape[0], img_np.shape[1])

        # Preprocess
        img_np, template_np, mask_np = self._resize_triplet(img_np, template_np, mask_np)
        if self.enable_augment:
            img_np, template_np, mask_np = self._augment_sample(img_np, template_np, mask_np)

        sample: Dict[str, Any] = {
            "image": self._to_tensor(img_np),
            "template": self._to_tensor(template_np),
            "meta": {
                "image_path": record["image_path"],
                "template_path": record["template_path"],
                "gt_path": gt_path,
                "orig_size": orig_size,
                "resized_size": (self.input_size, self.input_size),
            },
        }
        if mask_np is not None:
            sample["gt"] = self._mask_to_tensor(mask_np)

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch of samples."""
    images = torch.stack([sample["image"] for sample in batch])
    templates = torch.stack([sample["template"] for sample in batch])

    gt = None
    if "gt" in batch[0]:
        gt = torch.stack([sample["gt"] for sample in batch])

    return {
        "image": images,
        "template": templates,
        "gt": gt,
        "meta": [sample["meta"] for sample in batch],
    }


__all__ = [
    "CODDataset",
    "collate_fn",
    "INPUT_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
