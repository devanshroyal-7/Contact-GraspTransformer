from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from checkpoint_io import load_training_checkpoint, save_training_checkpoint
from data.dataset import CGNDataset, DatasetConfig, resolve_train_cap
from loss import CGNLoss
from models.model import ContactGraspNet

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    """Concrete training configuration built once from argparse / W&B."""

    data_dir: str
    backbone: str
    cpe_mode: str
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    scheduler: str
    scheduler_gamma: float
    grad_clip_max_norm: float
    loss_adds_weight: float
    loss_width_weight: float
    num_points: int
    overfit_one_batch: bool
    manifest: str
    budgets: str
    train_objects_per_category: Optional[int]
    budget_preset: Optional[str]
    checkpoint_dir: str
    resume: Optional[str]
    save_every: int
    wandb_project: str
    wandb_entity: str
    wandb_mode: str
    checkpoint_run_folder: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        values = {name: mapping[name] for name in known if name in mapping}
        return cls(**values)


def _mapping_from_args_and_wandb(args: argparse.Namespace, wb_cfg=None) -> dict[str, Any]:
    """Build a plain dict from CLI args, overlaying W&B sweep overrides when present."""
    raw = dict(vars(args))
    if wb_cfg is None:
        return raw
    try:
        overlay = dict(wb_cfg)
    except (TypeError, ValueError):
        overlay = {k: getattr(wb_cfg, k) for k in raw if hasattr(wb_cfg, k)}
    raw.update({k: overlay[k] for k in raw if k in overlay})
    return raw


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
    if cfg.scheduler == "step":
        step_size = max(1, cfg.epochs // 3)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=cfg.scheduler_gamma
        )
    if cfg.scheduler == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler_gamma, patience=5
        )
    return None


def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader,
    device: torch.device | str,
    *,
    split: str = "val",
) -> dict[str, float]:
    """Evaluate ``model`` on ``val_loader``; keys use ``{split}/…`` prefix."""
    model.eval()
    total_loss = 0.0
    total_conf = 0.0
    total_adds = 0.0
    total_width = 0.0
    with torch.no_grad():
        for batch in val_loader:
            points = batch["points"].to(device)
            targets = {k: v.to(device) for k, v in batch.items()}
            preds = model(points)
            loss_dict = criterion(preds, targets)
            total_loss += loss_dict["loss"].item()
            total_conf += loss_dict["l_conf"].item()
            total_adds += loss_dict["l_adds"].item()
            total_width += loss_dict["l_width"].item()
    n = max(len(val_loader), 1)
    return {
        f"{split}/loss": total_loss / n,
        f"{split}/loss_conf": total_conf / n,
        f"{split}/loss_adds": total_adds / n,
        f"{split}/loss_width": total_width / n,
    }


def save_checkpoint(
    path: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_val_loss: float,
    cfg: TrainConfig,
) -> None:
    # ``asdict(cfg)`` persists backbone + cpe_mode for inference rebuild.
    save_training_checkpoint(
        path, epoch, model, optimizer, scheduler, best_val_loss, asdict(cfg),
    )


def build_dataloaders(
    cfg: TrainConfig,
    run=None,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train/val/test loaders from real data, or mock data if missing."""
    if (os.path.exists(cfg.data_dir) and len(os.listdir(cfg.data_dir)) > 0
            and os.path.exists(cfg.manifest)):
        active_cap = resolve_train_cap(
            cfg.budgets,
            override=cfg.train_objects_per_category,
            preset=cfg.budget_preset,
        )
        if run is not None:
            wandb.config.update({"active_train_objects_per_category": active_cap},
                                allow_val_change=True)
        print(f"Training cap: {active_cap} object(s) per category")

        ds_cfg = DatasetConfig.from_paths(
            data_dir=cfg.data_dir,
            manifest_path=cfg.manifest,
            budget_path=cfg.budgets,
            num_points=cfg.num_points,
            train_objects_per_category=cfg.train_objects_per_category,
            budget_preset=cfg.budget_preset,
        )
        train_dataset = CGNDataset(ds_cfg, split="train")
        val_dataset = CGNDataset(ds_cfg, split="val")
        test_dataset = CGNDataset(ds_cfg, split="test")
        print(f"Datasets -> train={len(train_dataset)}  "
              f"val={len(val_dataset)}  test={len(test_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False)
        test_loader = (DataLoader(test_dataset, batch_size=cfg.batch_size,
                                   shuffle=False)
                       if len(test_dataset) > 0 else None)
        return train_loader, val_loader, test_loader

    print("No real data found. Using mock random data to verify pipeline builds.")
    n = cfg.num_points
    mock_data = [
        {
            "points": torch.randn(n, 3),
            "confidence": torch.randint(0, 2, (n,)).float(),
            "approach_dirs": torch.randn(n, 3),
            "base_dirs": torch.randn(n, 3),
            "widths": torch.rand(n) * 0.1,
        }
        for _ in range(8)
    ]
    train_loader = DataLoader(mock_data, batch_size=cfg.batch_size)
    return train_loader, train_loader, None


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Run one training epoch; return averaged train metrics dict."""
    model.train()
    total_loss = 0.0
    total_conf = 0.0
    total_adds = 0.0
    total_width = 0.0
    batches_seen = 0

    for batch in train_loader:
        points = batch["points"].to(device)
        targets = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        preds = model(points)

        loss_dict = criterion(preds, targets)
        loss = loss_dict["loss"]

        loss.backward()

        if cfg.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)

        optimizer.step()

        total_loss += loss.item()
        total_conf += loss_dict["l_conf"].item()
        total_adds += loss_dict["l_adds"].item()
        total_width += loss_dict["l_width"].item()
        batches_seen += 1

        if cfg.overfit_one_batch:
            break

    n_batches = max(batches_seen, 1)
    return {
        "train/loss": total_loss / n_batches,
        "train/loss_conf": total_conf / n_batches,
        "train/loss_adds": total_adds / n_batches,
        "train/loss_width": total_width / n_batches,
        "train/lr": optimizer.param_groups[0]["lr"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/out", help="Path to datasets")
    parser.add_argument("--backbone", type=str, default="ptv3", choices=["pn2", "ptv3"], help="Backbone type")
    parser.add_argument(
        "--cpe_mode",
        type=str,
        default="sparse3d",
        choices=["knn", "conv1d", "sparse3d"],
        help="PTv3 xCPE (conditional positional encoding); ignored when backbone is pn2",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step", "reduce_lr_on_plateau"])
    parser.add_argument("--scheduler_gamma", type=float, default=0.3)
    parser.add_argument("--grad_clip_max_norm", type=float, default=0.0, help="0 disables gradient clipping")
    parser.add_argument("--loss_adds_weight", type=float, default=10.0)
    parser.add_argument("--loss_width_weight", type=float, default=1.0)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--overfit_one_batch", action="store_true", help="Test flag")
    parser.add_argument("--manifest", type=str,
                        default="data/acronym/manifest.json",
                        help="Path to manifest.json")
    parser.add_argument("--budgets", type=str,
                        default="data/acronym/training_budgets.json",
                        help="Path to training_budgets.json")
    parser.add_argument("--train_objects_per_category", type=int, default=None,
                        help="Override the budget preset (cap on train meshes per category)")
    parser.add_argument("--budget_preset", type=str, default=None,
                        help="Named preset from training_budgets.json "
                             "(overrides active_preset; e.g. 1_per_cat, 2_per_cat, 5_per_cat, 10_per_cat)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory where training checkpoints are saved")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save an epoch checkpoint every N epochs; 0 disables per-epoch snapshots")
    parser.add_argument("--wandb_project", type=str, default="cgn-sweep",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="cgn-transformer",
                        help="Weights & Biases entity/team name")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="Weights & Biases mode")
    return parser.parse_args(argv)


def _fmt_run_value(v: object) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def prepare_run(cfg: TrainConfig, run=None) -> str:
    """Stamp checkpoint paths / W&B metadata; return the human run name."""
    gc = cfg.grad_clip_max_norm or 0
    gc_tag = f"gc{_fmt_run_value(gc)}" if gc > 0 else "gcOff"
    backbone_s = str(cfg.backbone)

    name_parts = [backbone_s]
    if backbone_s == "ptv3":
        name_parts.append(str(cfg.cpe_mode or "knn"))
    name_parts.extend([
        f"bs{cfg.batch_size}",
        f"lr{_fmt_run_value(cfg.lr)}",
        gc_tag,
    ])
    run_name = "_".join(name_parts)
    run_folder = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(str(cfg.checkpoint_dir), backbone_s, run_folder)
    cfg.checkpoint_dir = checkpoint_dir
    cfg.checkpoint_run_folder = run_folder
    if run is not None:
        wandb.config.update(
            {"checkpoint_dir": checkpoint_dir, "checkpoint_run_folder": run_folder},
            allow_val_change=True,
        )
        run.name = run_name
        wb_tags = [f"backbone:{backbone_s}"]
        if backbone_s == "ptv3":
            wb_tags.append(f"cpe_mode:{cfg.cpe_mode or 'knn'}")
        run.tags = list(run.tags or []) + wb_tags
        print(f"W&B run name: {run_name}")
    return run_name


def run_training(cfg: TrainConfig, run=None) -> None:
    """Build model/opt, optionally resume, and run the train/val(/test) loop."""
    backbone_s = str(cfg.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpe_mode = None
    if backbone_s == "ptv3":
        cpe_mode = str(cfg.cpe_mode or "knn")
        print(f"Using {cfg.backbone} backbone (cpe_mode={cpe_mode}) on {device}")
    else:
        print(f"Using {cfg.backbone} backbone on {device}")

    model = ContactGraspNet(
        backbone=cfg.backbone,
        cpe_mode=cpe_mode,
    ).to(device)
    criterion = CGNLoss(
        adds_weight=cfg.loss_adds_weight,
        width_weight=cfg.loss_width_weight,
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to {cfg.checkpoint_dir}")

    start_epoch = 0
    best_val_loss = float("inf")

    if cfg.resume:
        _, start_epoch, best_val_loss = load_training_checkpoint(
            cfg.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(
            f"Resumed from {cfg.resume} at epoch {start_epoch} "
            f"(best val loss: {best_val_loss:.4f})"
        )

    train_loader, val_loader, test_loader = build_dataloaders(cfg, run=run)

    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, cfg
        )
        if cfg.overfit_one_batch:
            print(
                f"Epoch {epoch} | Loss: {train_metrics['train/loss']:.4f} | "
                f"Conf: {train_metrics['train/loss_conf']:.4f}"
            )

        metrics = {**train_metrics, "epoch": epoch}

        val_metrics = evaluate(model, criterion, val_loader, device)
        metrics.update(val_metrics)
        if run is not None:
            wandb.log(metrics)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val/loss"])
            else:
                scheduler.step()

        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pt")
            save_checkpoint(
                best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )
            print(
                f"Saved new best checkpoint to {best_ckpt_path} "
                f"(val loss: {best_val_loss:.4f})"
            )

        last_ckpt_path = os.path.join(cfg.checkpoint_dir, "last.pt")
        save_checkpoint(
            last_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        if cfg.save_every > 0 and ((epoch + 1) % cfg.save_every == 0):
            epoch_ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch + 1:03d}.pt")
            save_checkpoint(
                epoch_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )

        if not cfg.overfit_one_batch:
            print(
                f"Epoch {epoch} | Train: {train_metrics['train/loss']:.4f} "
                f"| Val: {val_metrics['val/loss']:.4f}"
            )

    if test_loader is not None:
        test_metrics = evaluate(
            model, criterion, test_loader, device, split="test"
        )
        if run is not None:
            wandb.log(test_metrics)
        print(
            "Test set (held-out meshes): "
            f"loss={test_metrics['test/loss']:.4f}"
        )

    print("Training finished.")
    if run is not None:
        run.finish()


def main():
    args = parse_args()

    if args.wandb_mode != "disabled" and wandb is None:
        raise ImportError(
            "wandb is not installed. Install it or run with --wandb_mode disabled."
        )

    if wandb is not None and args.wandb_mode != "disabled":
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=vars(args),
            mode=args.wandb_mode,
        )
        cfg = TrainConfig.from_mapping(_mapping_from_args_and_wandb(args, wandb.config))
    else:
        run = None
        cfg = TrainConfig.from_mapping(_mapping_from_args_and_wandb(args))

    prepare_run(cfg, run=run)
    run_training(cfg, run=run)


if __name__ == "__main__":
    main()
