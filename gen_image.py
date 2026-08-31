"""
Stage-I mental-template generation for CVI-COD.

For each input camouflage image, this script uses the frozen image-editing
foundation model Qwen-Image-Edit-2509 to synthesize a target-revealing
"mental template". Templates are produced solely from the input image and the
fixed instruction prompt; ground-truth masks are never used.

Generation configuration (as used in the paper):
    model              : Qwen/Qwen-Image-Edit-2509  (frozen, no fine-tuning)
    prompt             : the fixed instruction in EDIT_PROMPT below
    num_inference_steps: 40
    true_cfg_scale     : 4.0
    guidance_scale     : 1.0
    negative_prompt    : " "
    seed               : per-image deterministic seed = base_seed (42) + hash(relative path),
                         so templates are reproducible and the run is resume-safe.

Output templates mirror the relative directory structure of the input images,
saved as PNG with the same stem, so Stage-II can match image/template pairs.

Usage:
    python gen_image.py --data-dir <images> --output-dir <templates>
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


# ----- Default configuration -----
DATA_DIR = "./images/"
EDITED_DIR = "./templates/"
MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
BASE_SEED = 42
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 1.0
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = " "

EDIT_PROMPT = (
    "Reveal the camouflaged animal clearly by removing the surrounding background. "
    "If it is hidden or covered, reduce the covering elements so the animal becomes fully visible. "
    "Enhance only the animal's colors and contrast while keeping the animal's shape and position unchanged overall."
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_global_seed(seed: int) -> None:
    """Fix common RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(data_dir: Path) -> List[Path]:
    """Recursively list image files under a directory."""
    return sorted(p.resolve() for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def make_generator(key: str, device: str, base_seed: int) -> torch.Generator:
    """Deterministic per-image generator: seed = base_seed + hash(key)."""
    hashed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return torch.Generator(device=device).manual_seed(base_seed + hashed)


def prepare_pipeline(model_id: str, device: str, disable_progress_bar: bool) -> QwenImageEditPlusPipeline:
    """Load the Qwen-Image-Edit pipeline (bfloat16 on GPU, float32 on CPU)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipeline.set_progress_bar_config(disable=disable_progress_bar)
    return pipeline


def run_image_edit(pipeline: QwenImageEditPlusPipeline, image: Image.Image,
                   generator: torch.Generator, prompt: str, true_cfg_scale: float,
                   num_inference_steps: int, guidance_scale: float,
                   negative_prompt: str) -> Image.Image:
    """Edit a single image and return the generated template."""
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
    return output.images[0]


def main() -> None:
    args = parse_cli_args()
    set_global_seed(args.base_seed)

    data_root = Path(args.data_dir).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    edited_dir = Path(args.output_dir)
    if args.run_name:
        edited_dir = edited_dir / args.run_name
    edited_dir = edited_dir.resolve()
    edited_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(data_root)
    if not images:
        raise FileNotFoundError(f"No images found under {data_root}")

    print(f"[gen_image] found {len(images)} images")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = prepare_pipeline(args.model_id, device, args.disable_progress_bar)

    for img_path in images:
        rel = img_path.relative_to(data_root)
        out_path = edited_dir / rel.with_suffix(".png")
        if out_path.exists():  # resume-safe: skip already-generated templates
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(img_path).convert("RGB")
        generator = make_generator(rel.with_suffix("").as_posix(), device, args.base_seed)
        template = run_image_edit(
            pipeline, image, generator, args.edit_prompt, args.true_cfg_scale,
            args.num_inference_steps, args.guidance_scale, args.negative_prompt,
        )
        template.save(out_path)
        print(f"[gen_image] saved {out_path.relative_to(edited_dir)}")

    print(f"[gen_image] done, templates under {edited_dir}")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-I template generation with Qwen-Image-Edit")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="input camouflage images")
    parser.add_argument("--output-dir", type=str, default=EDITED_DIR, help="output directory for templates")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Qwen-Image-Edit model id or local path")
    parser.add_argument("--run-name", type=str, default=None, help="optional output subdirectory")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--true-cfg-scale", type=float, default=TRUE_CFG_SCALE)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--edit-prompt", type=str, default=EDIT_PROMPT)
    parser.add_argument("--disable-progress-bar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
