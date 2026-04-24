import os
import torch
import argparse
from types import SimpleNamespace
from torch.utils.data import DataLoader
from data.dataset import CGNDataset, resolve_train_cap
from models.model import ContactGraspNet
from loss import CGNLoss

try:
    import wandb
except ImportError:
    wandb = None


def build_optimizer(model, cfg):
    params = model.parameters()
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg):
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


def evaluate(model, criterion, val_loader, device):
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
        "val/loss": total_loss / n,
        "val/loss_conf": total_conf / n,
        "val/loss_adds": total_adds / n,
        "val/loss_width": total_width / n,
    }


def _cfg_to_dict(cfg):
    """Serialize ``wandb.config`` / ``SimpleNamespace`` / argparse.Namespace."""
    if hasattr(cfg, "as_dict"):
        try:
            return dict(cfg.as_dict())
        except Exception:
            pass
    try:
        return dict(cfg)
    except TypeError:
        return dict(vars(cfg))


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss, cfg):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": _cfg_to_dict(cfg),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return checkpoint, start_epoch, best_val_loss


def main():
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
    args = parser.parse_args()

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
        cfg = wandb.config
    else:
        run = None
        cfg = SimpleNamespace(**vars(args))

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    def _cfg_get(name, default=None):
        try:
            return cfg[name]
        except (KeyError, AttributeError):
            return getattr(cfg, name, default)

    gc = _cfg_get("grad_clip_max_norm", 0) or 0
    gc_tag = f"gc{_fmt(gc)}" if gc > 0 else "gcOff"
    backbone_s = str(_cfg_get("backbone"))
    name_parts = [backbone_s]
    if backbone_s == "ptv3":
        name_parts.append(str(_cfg_get("cpe_mode", "knn")))
    name_parts.extend([
        f"bs{_cfg_get('batch_size')}",
        f"lr{_fmt(_cfg_get('lr'))}",
        gc_tag,
    ])
    run_name = "_".join(name_parts)

    if run is not None:
        run.name = run_name
        wb_tags = [f"backbone:{backbone_s}"]
        if backbone_s == "ptv3":
            wb_tags.append(f"cpe_mode:{_cfg_get('cpe_mode', 'knn')}")
        run.tags = list(run.tags or []) + wb_tags
        print(f"W&B run name: {run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_kwargs = None
    if backbone_s == "ptv3":
        backbone_kwargs = {"cpe_mode": str(_cfg_get("cpe_mode", "knn"))}
        print(f"Using {cfg.backbone} backbone (cpe_mode={backbone_kwargs['cpe_mode']}) on {device}")
    else:
        print(f"Using {cfg.backbone} backbone on {device}")

    model = ContactGraspNet(
        backbone_type=cfg.backbone,
        backbone_kwargs=backbone_kwargs,
    ).to(device)
    criterion = CGNLoss(
        adds_weight=cfg.loss_adds_weight,
        width_weight=cfg.loss_width_weight,
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")

    if cfg.resume:
        _, start_epoch, best_val_loss = load_checkpoint(
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

    if (os.path.exists(cfg.data_dir) and len(os.listdir(cfg.data_dir)) > 0
            and os.path.exists(cfg.manifest)):
        try:
            active_cap = resolve_train_cap(
                cfg.budgets,
                override=cfg.train_objects_per_category,
                preset=cfg.budget_preset,
            )
            if run is not None:
                wandb.config.update({"active_train_objects_per_category": active_cap},
                                    allow_val_change=True)
            print(f"Training cap: {active_cap} object(s) per category")
        except Exception as e:
            print(f"Warning: could not resolve training budget ({e})")

        ds_kwargs = dict(
            data_dir=cfg.data_dir,
            manifest_path=cfg.manifest,
            budget_path=cfg.budgets,
            num_points=cfg.num_points,
            train_objects_per_category=cfg.train_objects_per_category,
            budget_preset=cfg.budget_preset,
        )
        train_dataset = CGNDataset(split="train", **ds_kwargs)
        val_dataset   = CGNDataset(split="val",   **ds_kwargs)
        test_dataset  = CGNDataset(split="test",
                                    data_dir=cfg.data_dir,
                                    manifest_path=cfg.manifest,
                                    budget_path=cfg.budgets,
                                    num_points=cfg.num_points)
        print(f"Datasets -> train={len(train_dataset)}  "
              f"val={len(val_dataset)}  test={len(test_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False)
        test_loader = (DataLoader(test_dataset, batch_size=cfg.batch_size,
                                   shuffle=False)
                       if len(test_dataset) > 0 else None)
    else:
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
        val_loader = train_loader
        test_loader = None

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total_loss = 0
        total_conf = 0
        total_adds = 0
        total_width = 0

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

            if args.overfit_one_batch:
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} | Conf: {loss_dict['l_conf'].item():.4f}"
                )
                break

        n_batches = max(len(train_loader), 1)
        metrics = {
            "train/loss": total_loss / n_batches,
            "train/loss_conf": total_conf / n_batches,
            "train/loss_adds": total_adds / n_batches,
            "train/loss_width": total_width / n_batches,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }

        val_metrics = evaluate(model, criterion, val_loader, device)
        metrics.update(val_metrics)
        if run is not None:
            wandb.log(metrics)

        if scheduler is not None:
            # If using ReduceLROnPlateau, you MUST provide the metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val/loss'])
            else:
                scheduler.step()

        is_best = val_metrics["val/loss"] < best_val_loss
        if is_best:
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

        if not args.overfit_one_batch:
            print(f"Epoch {epoch} | Train: {total_loss / n_batches:.4f} "
                  f"| Val: {val_metrics['val/loss']:.4f}")

    if test_loader is not None:
        test_metrics = evaluate(model, criterion, test_loader, device)
        test_metrics = {k.replace("val/", "test/"): v for k, v in test_metrics.items()}
        if run is not None:
            wandb.log(test_metrics)
        print("Test set (held-out meshes): "
              f"loss={test_metrics['test/loss']:.4f}")

    print("Training finished.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
