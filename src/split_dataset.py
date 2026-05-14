from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import prepare_dataset
from project_paths import from_project_root, resolve_cli_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a CT dataset and create train/val/test split manifests.")
    parser.add_argument("--dataset-root", type=Path, default=from_project_root("Dataset"))
    parser.add_argument("--split-dir", type=Path, default=from_project_root("Dataset", "splits"))
    parser.add_argument("--results-dir", type=Path, default=from_project_root("results"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recreate split CSV files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_cli_path(args.dataset_root)
    args.split_dir = resolve_cli_path(args.split_dir)
    args.results_dir = resolve_cli_path(args.results_dir)

    prepared = prepare_dataset(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        results_dir=args.results_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force,
    )

    analysis = prepared["analysis"]
    metadata = prepared["metadata"]
    summary = {
        "layout": analysis["layout"],
        "total_images": analysis["total_images"],
        "class_counts": analysis["class_counts"],
        "split_counts": metadata["split_counts"],
        "missing_files": len(analysis["missing_files"]),
        "broken_files": len(analysis["broken_files"]),
        "analysis_report": prepared["analysis_txt"],
        "split_metadata": prepared["metadata_path"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
