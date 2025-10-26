"""Extract zip archives under an OmniRetarget dataset folder and load all .npz files.

Usage:
    python load.py <dataset_dir>

This will:
    - extract any .zip files found directly inside <dataset_dir> into
        a sibling directory named <zipname>_extracted (if not already extracted),
    - recursively find all .npz files under <dataset_dir> (including extracted folders),
    - print a short summary for each .npz (qpos shape and fps if present).
"""

import sys
import zipfile
from pathlib import Path
import numpy as np


def extract_zips(dataset_dir: Path) -> None:
    """Extract all top-level .zip files in dataset_dir if not already extracted."""
    for z in dataset_dir.glob("*.zip"):
        target_dir = dataset_dir / f"{z.stem}_extracted"
        if target_dir.exists():
            print(f"Skipping extraction (already exists): {z.name} -> {target_dir.name}")
            continue
        print(f"Extracting {z.name} -> {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(target_dir)


def find_npz_files(dataset_dir: Path):
    return sorted(dataset_dir.rglob("*.npz"))


def inspect_npz(npz_path: Path):
    try:
        with np.load(npz_path) as data:
            info = {}
            if "qpos" in data:
                info["qpos.shape"] = data["qpos"].shape
            if "fps" in data:
                info["fps"] = float(data["fps"])
            # include other useful keys lightly
            for k in ("actions", "qvel", "qacc"):
                if k in data:
                    info[k + ".shape"] = data[k].shape
            return info
    except Exception as e:
        return {"error": str(e)}


def main(argv):
    if len(argv) < 2:
        print("Usage: python load.py <OmniRetarget_Dataset_dir>")
        return 1

    dataset_dir = Path(argv[1])
    if not dataset_dir.exists():
        print(f"Dataset dir not found: {dataset_dir}")
        return 1

    # Extract top-level zips
    extract_zips(dataset_dir)

    # Find all npz files underneath
    npz_files = find_npz_files(dataset_dir)
    if not npz_files:
        print("No .npz files found under dataset dir.")
        return 0

    print(f"Found {len(npz_files)} .npz files. Inspecting...")
    for p in npz_files:
        info = inspect_npz(p)
        print(f"{p.relative_to(dataset_dir)}: {info}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

