"""Print a lightweight ContactGraspNet model summary."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn

from inference import DEFAULT_NUM_POINTS, GraspPredictor


def _shape_of(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_shape_of(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shape_of(v) for v in value) + "]"
    return type(value).__name__


def _module_param_counts(module: nn.Module) -> tuple[int, int]:
    params = list(module.parameters(recurse=False))
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    return total, trainable


def print_flat_summary(model: nn.Module, sample: torch.Tensor) -> None:
    rows: list[tuple[str, str, str, int, int]] = []
    hooks = []

    for name, module in model.named_modules():
        if name == "" or any(module.children()):
            continue

        def hook(mod: nn.Module, _inputs: tuple[Any, ...], output: Any, module_name: str = name) -> None:
            total, trainable = _module_param_counts(mod)
            rows.append((module_name, mod.__class__.__name__, _shape_of(output), total, trainable))

        hooks.append(module.register_forward_hook(hook))

    try:
        model.eval()
        with torch.no_grad():
            model(sample)
    finally:
        for hook in hooks:
            hook.remove()

    headers = ("Module", "Type", "Output Shape", "Params", "Trainable")
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * width for width in widths)))
    for row in rows:
        print(fmt.format(*row))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Input shape: {tuple(sample.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ContactGraspNet model summary")
    parser.add_argument("--ckpt", required=True, help="Path to trained .pt file")
    parser.add_argument(
        "--summary",
        default="flat",
        choices=["flat"],
        help="Summary style to print",
    )
    parser.add_argument(
        "--backbone",
        default="ptv3",
        choices=["pn2", "ptv3"],
        help="Fallback backbone if the checkpoint has no embedded train config",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Fallback N when the checkpoint has no embedded num_points",
    )
    parser.add_argument(
        "--cpe-mode",
        default=None,
        choices=["knn", "conv1d", "sparse3d"],
        help="Override PTv3 cpe_mode (auto-detected from weights when possible)",
    )
    parser.add_argument("--device", default=None, help="'cuda', 'cpu' (default: cuda if available)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = GraspPredictor(
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        num_points=args.num_points,
        device=args.device,
        cpe_mode=args.cpe_mode,
    )
    sample = torch.zeros((1, predictor.num_points, 3), dtype=torch.float32, device=predictor.device)

    if args.summary == "flat":
        print_flat_summary(predictor.model, sample)


if __name__ == "__main__":
    main()
