"""
Testing/Evaluation script for CVINet.

Metrics are computed at the network input resolution (--img-size, with ground
truth resized accordingly); predicted masks are saved at the original image
resolution. When --gt-dir is omitted, only predictions are saved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from datasets import INPUT_SIZE, CODDataset, collate_fn
from metric.metric_recorder import MetricRecorder
from model import CVI_Net

# Default configuration
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_THRESHOLD = 0.5
DEFAULT_OUTPUT_DIR = Path("./test_outputs")
DEFAULT_BACKBONE = "pvt_v2_b4"
DEFAULT_IMG_SIZE = INPUT_SIZE


def load_weights(model: CVI_Net, checkpoint_path: Path, map_location: torch.device) -> None:
    """Load model weights from checkpoint."""
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model_state"])
    print(f"[Checkpoint] Loaded weights from {checkpoint_path}")


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Save binary mask as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


def evaluate(
    model: CVI_Net,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: Path,
    compute_metrics: bool,
) -> Optional[Dict[str, Dict]]:
    """Evaluate model, save predictions, and return metrics (None without GT)."""
    model.eval()
    metric = MetricRecorder()
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device)
            templates = batch["template"].to(device)
            gt_batch = batch["gt"]
            meta_batch = batch["meta"]

            prob = torch.sigmoid(model(images, templates))

            for b in range(prob.shape[0]):
                sample_count += 1
                meta = meta_batch[b]
                original_size = tuple(meta["orig_size"])

                prob_cur = prob[b:b+1]

                # Compute metrics at input resolution if GT available
                if compute_metrics:
                    prob_np_metric = (prob_cur.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    gt_np_metric = (gt_batch[b].squeeze().cpu().numpy() * 255).astype(np.uint8)
                    metric.update(prob_np_metric, gt_np_metric)

                # Save prediction at original size
                prob_save = prob_cur
                if prob_save.shape[-2:] != original_size:
                    prob_save = F.interpolate(prob_save, size=original_size, mode="bilinear", align_corners=False)
                pred_np = ((prob_save > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255)

                pred_name = Path(meta["image_path"]).stem + ".png"
                save_mask(pred_np, preds_dir / pred_name)

            if idx % 20 == 0:
                print(f"[Eval] Processed {sample_count} samples")

    print(f"[Eval] Predictions saved to {preds_dir}")
    if not compute_metrics:
        return None

    results = metric.show()
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Eval] Metrics saved to {metrics_path}")
    return results


def create_run_dir(base_dir: Path, run_name: Optional[str]) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        candidate = base_dir / run_name
    else:
        candidate = base_dir / datetime.now().strftime("run_%Y%m%d-%H%M%S")
    original = candidate
    counter = 1
    while candidate.exists():
        candidate = original.parent / f"{original.name}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVINet Testing")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing test images")
    parser.add_argument("--template-dir", type=Path, required=True, help="Directory containing template images")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Directory containing ground truth masks (optional)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Binarization threshold")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None, help="Output subdirectory name")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = create_run_dir(args.output_dir, args.run_name)
    print(f"[Eval] Outputs will be saved under: {run_dir}")

    # Create dataset
    dataset = CODDataset(
        image_dir=args.image_dir,
        template_dir=args.template_dir,
        gt_dir=args.gt_dir,
        input_size=args.img_size,
        enable_augment=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    print(f"[Eval] Loaded {len(dataset)} samples at {args.img_size}x{args.img_size}")

    # Create and load model
    model = CVI_Net(
        pretrained_backbone=not args.no_pretrained_backbone,
        backbone_name=args.backbone,
    ).to(device)
    load_weights(model, args.checkpoint, map_location=device)

    # Save config
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Evaluate
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        output_dir=run_dir,
        compute_metrics=args.gt_dir is not None,
    )
    if results is not None:
        print(f"[Eval] Final metrics: {json.dumps(results['numerical'], indent=2)}")


if __name__ == "__main__":
    main()
