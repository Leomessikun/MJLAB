#!/usr/bin/env python3
"""
Batch-convert OmniRetarget NPZ files (qpos/fps) into mjlab motion.npz files.

Example:
  uv run python convert_omniretarget_batch.py \
    --input-dir OmniRetarget_Dataset/robot-terrain \
    --output-dir OmniRetarget_Dataset/converted \
    --output-fps 50 --device cpu
"""
from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

from mjlab.scripts.omniretarget_npz_to_npz import convert


def _convert_one(args: tuple[Path, Path, float, str]) -> tuple[Path, bool, str]:
    in_path, out_path, fps, device = args
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert(str(in_path), str(out_path), output_fps=fps, device=device)
        return out_path, True, "ok"
    except Exception as e:
        return out_path, False, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing OmniRetarget .npz files")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write converted mjlab motion files")
    p.add_argument("--output-fps", type=float, default=50.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers for conversion")
    p.add_argument("--glob", type=str, default="*.npz", help="Glob pattern to select inputs")
    args = p.parse_args()

    inputs = sorted(args.input_dir.rglob(args.glob))
    if not inputs:
        print(f"No files found in {args.input_dir} matching {args.glob}")
        return

    tasks: list[tuple[Path, Path, float, str]] = []
    for in_path in inputs:
        out_name = f"motion_{in_path.stem}.npz"
        out_path = args.output_dir / out_name
        if out_path.exists():
            continue
        tasks.append((in_path, out_path, args.output_fps, args.device))

    print(f"Found {len(inputs)} inputs; will convert {len(tasks)} new files to {args.output_dir}")

    if args.workers <= 1:
        for t in tasks:
            out, ok, msg = _convert_one(t)
            print(("[OK]" if ok else "[FAIL]"), out, ("" if ok else msg))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for out, ok, msg in ex.map(_convert_one, tasks):
                print(("[OK]" if ok else "[FAIL]"), out, ("" if ok else msg))


if __name__ == "__main__":
    main()
