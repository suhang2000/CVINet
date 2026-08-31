"""
Training script for CVINet.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import amp, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import INPUT_SIZE, CODDataset, collate_fn
from model import CVI_Net

# Default configuration
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 20
DEFAULT_NUM_WORKERS = 8
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LOG_INTERVAL = 10
DEFAULT_OUTPUT_DIR = Path("./checkpoints")
DEFAULT_SCHEDULER = "cosine"
DEFAULT_CHECKPOINT_INTERVAL = 50
DEFAULT_GRAD_ACCUM = 1
DEFAULT_SEED = 42
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_MIN_LR_RATIO = 0.2
DEFAULT_BACKBONE = "pvt_v2_b4"
DEFAULT_IMG_SIZE = INPUT_SIZE


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for segmentation."""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    numerator = 2 * (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    loss = 1 - numerator / denominator
    return loss.mean()


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    """DataLoader worker initialization for reproducibility."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train_one_epoch(
    model: CVI_Net,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    bce_loss_fn: nn.Module,
    epoch: int,
    log_interval: int,
    writer: SummaryWriter,
    global_step: int,
    scaler: amp.GradScaler,
    use_amp: bool,
    grad_accum: int,
    clip_grad: float,
) -> Tuple[float, int]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    optimizer.zero_grad(set_to_none=True)
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    progress_bar = tqdm(dataloader, desc=f"Train {epoch}", leave=False)

    for step, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        templates = batch["template"].to(device)
        gt = batch["gt"].to(device)

        with amp.autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(images, templates)
            loss = bce_loss_fn(pred, gt) + dice_loss(pred, gt)

        running_loss += loss.item()
        loss = loss / grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        writer.add_scalar("train/loss_step", loss.item(), global_step)
        global_step += 1

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / (step + 1)
            progress_bar.set_postfix_str(f"Loss: {avg_loss:.4f}")

    return running_loss / max(1, len(dataloader)), global_step


def save_checkpoint(
    output_dir: Path,
    model: CVI_Net,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
) -> Path:
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
    }
    ckpt_path = output_dir / f"epoch_{epoch}.pt"
    torch.save(state, ckpt_path)
    logging.getLogger("train").info(f"[Checkpoint] Saved -> {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVINet Training")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing original images")
    parser.add_argument("--template-dir", type=Path, required=True, help="Directory containing template images")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing ground truth masks")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None, help="Output subdirectory name")
    parser.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER, choices=["none", "cosine"])
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Mixed precision training (disable with --no-fp16)")
    parser.add_argument("--grad-accumulation", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--clip-grad", type=float, default=0, help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="Disable pretrained backbone")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--min-lr-ratio", type=float, default=DEFAULT_MIN_LR_RATIO)
    return parser.parse_args()


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


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
):
    if name == "cosine":
        warmup_epochs = max(0, warmup_epochs)
        schedulers = []
        milestones = []

        if warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0 / warmup_epochs,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
            )
            milestones.append(warmup_epochs)

        t_max = max(1, epochs - warmup_epochs)
        eta_min = max(1e-8, min_lr_ratio * optimizer.param_groups[0]["lr"])
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        )

        if warmup_epochs > 0:
            return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)
        return schedulers[0]
    return None


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    """Save run configuration to JSON."""
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg["command"] = " ".join(sys.argv)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Create dataset
    dataset = CODDataset(
        image_dir=args.image_dir,
        template_dir=args.template_dir,
        gt_dir=args.gt_dir,
        input_size=args.img_size,
        enable_augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=partial(seed_worker, base_seed=args.seed),
    )

    # Setup output directory
    if args.resume:
        run_dir = args.resume.parent
    else:
        run_dir = create_run_dir(args.output_dir, args.run_name)

    logger = setup_logger(run_dir)
    logger.info(f"[Train] Outputs will be saved under: {run_dir}")
    save_run_config(run_dir, args)

    writer = SummaryWriter(log_dir=run_dir / "tensorboard")
    logger.info(f"[Data] Train samples: {len(dataset)} | Input size: {args.img_size}x{args.img_size}")

    # Create model
    model = CVI_Net(
        pretrained_backbone=not args.no_pretrained_backbone,
        backbone_name=args.backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs, args.min_lr_ratio)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[Model] Parameters: {total_params / 1e6:.2f}M")

    start_epoch = 1
    use_amp = args.fp16 and torch.cuda.is_available()
    if args.fp16 and not torch.cuda.is_available():
        logger.warning("[Train] --fp16 ignored because CUDA is not available.")
    scaler = amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, args.grad_accumulation)
    clip_grad = max(0.0, args.clip_grad)

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"[Resume] Loaded checkpoint from {args.resume}, restarting at epoch {start_epoch}")

    global_step = 0

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss, global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            device=device,
            optimizer=optimizer,
            bce_loss_fn=bce_loss_fn,
            epoch=epoch,
            log_interval=args.log_interval,
            writer=writer,
            global_step=global_step,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum=grad_accum,
            clip_grad=clip_grad,
        )

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)
        logger.info(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f} | LR: {current_lr:.7f}")

        if scheduler is not None:
            scheduler.step()

        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(run_dir, model, optimizer, scheduler, scaler, epoch)

    # Save final checkpoint unless the last epoch was already saved above
    if args.epochs % args.checkpoint_interval != 0:
        save_checkpoint(run_dir, model, optimizer, scheduler, scaler, args.epochs)
    writer.close()
    logger.info(f"[Training Complete] Checkpoints stored in {run_dir}")


if __name__ == "__main__":
    main()
