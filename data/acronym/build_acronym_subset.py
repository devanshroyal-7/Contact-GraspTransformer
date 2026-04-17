"""Build the 15-category ACRONYM subset used by this project.

Deterministically copies 12 object meshes + grasp files per category
(10 train + 2 test) from the external ACRONYM checkout into
``data/acronym/`` and writes ``manifest.json`` annotated with a
``split`` and ``rank`` field for every object.

The manifest drives both ``data/generate_data.py`` (what to render)
and ``data/dataset.py`` (what to load at train/test time, respecting
the budget in ``training_budgets.json``).

Usage:
    python data/acronym/build_acronym_subset.py
    python data/acronym/build_acronym_subset.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

CATEGORIES: list[str] = [
    "Mug", "Bowl", "Bottle", "Cup", "Pan",
    "Book", "Vase", "CellPhone", "Pencil", "ToyFigure",
    "Knife", "FoodItem", "Camera", "SodaCan", "WineBottle",
]

DEFAULT_SRC = "/home/devansh/dev/contact_graspnet_pytorch/acronym"
DEFAULT_DST = "/home/devansh/dev/idl_proj_trial/data/acronym"

N_TRAIN = 10
N_TEST = 2
N_TOTAL = N_TRAIN + N_TEST

GRASP_RE = re.compile(r"^(?P<cat>[A-Za-z0-9]+)_(?P<hash>[0-9a-fA-F]+)_(?P<scale>[0-9.eE+\-]+)\.h5$")


def find_pairs(src_root: Path, category: str) -> list[tuple[str, str, Path]]:
    """Return deterministically sorted (hash, scale_str, grasp_src_path) tuples
    for which both the .obj mesh and the grasp .h5 file exist.
    """
    mesh_dir = src_root / "meshes" / category
    grasp_dir = src_root / "grasps"
    if not mesh_dir.is_dir():
        return []

    obj_hashes = {p.stem for p in mesh_dir.iterdir() if p.suffix == ".obj"}

    by_hash: dict[str, tuple[str, str, Path]] = {}
    prefix = f"{category}_"
    for h5 in sorted(grasp_dir.glob(f"{prefix}*.h5")):
        m = GRASP_RE.match(h5.name)
        if not m or m.group("cat") != category:
            continue
        h = m.group("hash")
        if h not in obj_hashes:
            continue
        # Keep only the lexicographically-smallest scale per mesh_hash, so
        # each "object" in the manifest is a unique mesh. This matches the
        # original 5-object layout where every entry had a distinct hash.
        key = (h, m.group("scale"))
        if h not in by_hash or key < (h, by_hash[h][1]):
            by_hash[h] = (h, m.group("scale"), h5)

    pairs = sorted(by_hash.values(), key=lambda t: (t[0], t[1]))
    return pairs


def copy_if_needed(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """Copy ``src`` to ``dst`` unless ``dst`` already matches ``src`` by size.

    Returns True if the file was (or would be) copied.
    """
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build(src_root: Path, dst_root: Path, dry_run: bool = False) -> None:
    manifest: list[dict] = []
    per_cat_counts: list[tuple[str, int, int]] = []  # (cat, n_train, n_test)
    files_copied = 0
    files_skipped = 0
    missing: list[str] = []

    for cat in CATEGORIES:
        pairs = find_pairs(src_root, cat)
        if len(pairs) < N_TOTAL:
            missing.append(f"{cat} (only {len(pairs)} usable, need {N_TOTAL})")
            continue

        chosen = pairs[:N_TOTAL]

        n_train_done = 0
        n_test_done = 0
        for idx, (h, scale_str, grasp_src) in enumerate(chosen):
            if idx < N_TRAIN:
                split = "train"
                rank = idx + 1
                n_train_done += 1
            else:
                split = "test"
                rank = idx - N_TRAIN + 1
                n_test_done += 1

            obj_src = src_root / "meshes" / cat / f"{h}.obj"
            mtl_src = src_root / "meshes" / cat / f"{h}.mtl"
            obj_dst = dst_root / "meshes" / cat / f"{h}.obj"
            mtl_dst = dst_root / "meshes" / cat / f"{h}.mtl"
            grasp_dst = dst_root / "grasps" / grasp_src.name

            if copy_if_needed(obj_src, obj_dst, dry_run):
                files_copied += 1
            else:
                files_skipped += 1
            if mtl_src.exists():
                if copy_if_needed(mtl_src, mtl_dst, dry_run):
                    files_copied += 1
                else:
                    files_skipped += 1
            if copy_if_needed(grasp_src, grasp_dst, dry_run):
                files_copied += 1
            else:
                files_skipped += 1

            try:
                scale_val = float(scale_str)
            except ValueError:
                scale_val = 0.0

            manifest.append({
                "category": cat,
                "mesh_hash": h,
                "mesh_path": f"meshes/{cat}/{h}.obj",
                "grasp_file": grasp_src.name,
                "scale": scale_val,
                "split": split,
                "rank": rank,
            })

        per_cat_counts.append((cat, n_train_done, n_test_done))

    if missing:
        print("ERROR: the following categories do not have enough usable objects:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    manifest_path = dst_root / "manifest.json"
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Remove any mesh/grasp files that are not referenced by the new manifest
    # (e.g. leftovers from the 5-object layout or an earlier non-deduped run).
    kept_meshes: dict[str, set[str]] = {}
    kept_grasps: set[str] = set()
    for entry in manifest:
        kept_meshes.setdefault(entry["category"], set()).add(entry["mesh_hash"])
        kept_grasps.add(entry["grasp_file"])

    stale_removed = 0
    meshes_root = dst_root / "meshes"
    if meshes_root.is_dir():
        for cat_dir in meshes_root.iterdir():
            if not cat_dir.is_dir():
                continue
            keep = kept_meshes.get(cat_dir.name, set())
            if not keep:
                for f in cat_dir.iterdir():
                    if not dry_run:
                        f.unlink()
                    stale_removed += 1
                if not dry_run:
                    cat_dir.rmdir()
                continue
            for f in cat_dir.iterdir():
                if f.stem not in keep:
                    if not dry_run:
                        f.unlink()
                    stale_removed += 1

    grasps_root = dst_root / "grasps"
    if grasps_root.is_dir():
        for f in grasps_root.iterdir():
            if f.is_file() and f.name not in kept_grasps:
                if not dry_run:
                    f.unlink()
                stale_removed += 1

    print("=" * 60)
    print(f"{'Category':<14} {'Train':>6} {'Test':>6}")
    print("-" * 60)
    for cat, n_tr, n_te in per_cat_counts:
        print(f"{cat:<14} {n_tr:>6} {n_te:>6}")
    print("-" * 60)
    print(f"Total objects : {len(manifest)}  "
          f"(expected {len(CATEGORIES) * N_TOTAL})")
    print(f"Files copied  : {files_copied}")
    print(f"Files skipped : {files_skipped} (destination already up-to-date)")
    print(f"Stale removed : {stale_removed}")
    print(f"Manifest      : {manifest_path}{' (DRY RUN)' if dry_run else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="Path to the external ACRONYM checkout")
    parser.add_argument("--dst", default=DEFAULT_DST,
                        help="Path to the project-local data/acronym/ dir")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would happen without copying or writing")
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.is_dir():
        sys.exit(f"Source root not found: {src_root}")

    build(src_root, dst_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
