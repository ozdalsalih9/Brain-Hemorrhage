from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import load_split_metadata, prepare_dataset
from model_utils import (
    collect_predictions,
    compute_metrics,
    create_data_loaders,
    get_device,
    load_checkpoint,
    resolve_image_size,
    save_confusion_matrix,
    save_metrics,
)
from project_paths import from_project_root, resolve_cli_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained image classification checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=from_project_root("Dataset"))
    parser.add_argument("--split-dir", type=Path, default=from_project_root("Dataset", "splits"))
    parser.add_argument("--results-dir", type=Path, default=from_project_root("results"))
    parser.add_argument("--checkpoint", type=Path, default=from_project_root("models", "best_convnext_base_es_uint8.pth"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_cli_path(args.dataset_root)
    args.split_dir = resolve_cli_path(args.split_dir)
    args.results_dir = resolve_cli_path(args.results_dir)
    args.checkpoint = resolve_cli_path(args.checkpoint)

    run_name = args.checkpoint.stem.replace("best_", "")
    output_dir = Path(args.results_dir) / run_name
    prepare_dataset(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        results_dir=output_dir,
    )
    metadata = load_split_metadata(args.split_dir)
    class_names = metadata["class_names"]
    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model_name = checkpoint.get("model_name")
    image_size = resolve_image_size(checkpoint.get("image_size"), model_name)

    data_loaders = create_data_loaders(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        model_name=model_name,
        mean=checkpoint.get("normalization_mean"),
        std=checkpoint.get("normalization_std"),
    )
    y_true, y_pred = collect_predictions(model, data_loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    metrics_path = output_dir / "test_metrics.json"
    matrix_path = output_dir / "confusion_matrix_test.png"
    save_metrics(metrics, metrics_path)
    save_confusion_matrix(metrics, class_names, matrix_path, title=f"{run_name.title()} Confusion Matrix")

    print(json.dumps({"metrics_path": str(metrics_path), "confusion_matrix_path": str(matrix_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
