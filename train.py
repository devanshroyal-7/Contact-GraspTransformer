import os
import torch
import argparse
import wandb
from torch.utils.data import DataLoader
from data.dataset import CGNDataset
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
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--scheduler_gamma", type=float, default=0.3)
    parser.add_argument("--grad_clip_max_norm", type=float, default=0.0, help="0 disables gradient clipping")
    parser.add_argument("--loss_adds_weight", type=float, default=10.0)
    parser.add_argument("--loss_width_weight", type=float, default=1.0)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--overfit_one_batch", action="store_true", help="Test flag")
    args = parser.parse_args()

    run = wandb.init(project="cgn-sweep", config=vars(args))
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.backbone} backbone on {device}")

    model = ContactGraspNet(backbone_type=cfg.backbone).to(device)
    criterion = CGNLoss(
        adds_weight=cfg.loss_adds_weight,
        width_weight=cfg.loss_width_weight,
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if os.path.exists(cfg.data_dir) and len(os.listdir(cfg.data_dir)) > 0:
        train_dataset = CGNDataset(cfg.data_dir, num_points=cfg.num_points,
                                   split="train")
        val_dataset = CGNDataset(cfg.data_dir, num_points=cfg.num_points,
                                 split="val")
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False)
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

        if scheduler is not None:
            scheduler.step()

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

        if not args.overfit_one_batch:
            print(f"Epoch {epoch} | Train: {total_loss / n_batches:.4f} "
                  f"| Val: {val_metrics['val/loss']:.4f}")

    print("Training finished.")
    run.finish()


if __name__ == "__main__":
    main()
