import os
import torch
import argparse
import wandb
from torch.utils.data import DataLoader
from data.dataset import CGNDataset, resolve_train_cap
from models.model import ContactGraspNet
from loss import CGNLoss


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/out", help="Path to datasets")
    parser.add_argument("--backbone", type=str, default="pn2", choices=["pn2", "ptv3"], help="Backbone type")
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
    args = parser.parse_args()

    run = wandb.init(entity="cgn-transformer", project="cgn-sweep", config=vars(args))
    cfg = wandb.config

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
    run_name = "_".join([
        str(_cfg_get("backbone")),
        f"bs{_cfg_get('batch_size')}",
        f"lr{_fmt(_cfg_get('lr'))}",
        gc_tag,
    ])

    run.name = run_name
    run.tags = list(run.tags or []) + [f"backbone:{_cfg_get('backbone')}"]
    print(f"W&B run name: {run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.backbone} backbone on {device}")

    model = ContactGraspNet(backbone_type=cfg.backbone).to(device)
    criterion = CGNLoss(
        adds_weight=cfg.loss_adds_weight,
        width_weight=cfg.loss_width_weight,
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if (os.path.exists(cfg.data_dir) and len(os.listdir(cfg.data_dir)) > 0
            and os.path.exists(cfg.manifest)):
        try:
            active_cap = resolve_train_cap(
                cfg.budgets,
                override=cfg.train_objects_per_category,
                preset=cfg.budget_preset,
            )
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

    for epoch in range(cfg.epochs):
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
        wandb.log(metrics)

        if scheduler is not None:
            # If using ReduceLROnPlateau, you MUST provide the metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val/loss'])
            else:
                scheduler.step()

        if not args.overfit_one_batch:
            print(f"Epoch {epoch} | Train: {total_loss / n_batches:.4f} "
                  f"| Val: {val_metrics['val/loss']:.4f}")

    if test_loader is not None:
        test_metrics = evaluate(model, criterion, test_loader, device)
        test_metrics = {k.replace("val/", "test/"): v for k, v in test_metrics.items()}
        wandb.log(test_metrics)
        print("Test set (held-out meshes): "
              f"loss={test_metrics['test/loss']:.4f}")

    print("Training finished.")
    run.finish()


if __name__ == "__main__":
    main()
