from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from data_utils import load_split_metadata, prepare_dataset
from model_utils import (
    collect_predictions,
    compute_metrics,
    create_data_loaders,
    create_model_with_fallback,
    get_device,
    load_checkpoint,
    plot_training_history,
    resolve_image_size,
    save_confusion_matrix,
    save_history,
    save_metrics,
    train_model,
)
from project_paths import from_project_root, resolve_cli_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pretrained CNN on the CT classification dataset.")
    parser.add_argument("--dataset-root", type=Path, default=from_project_root("Dataset"))
    parser.add_argument("--split-dir", type=Path, default=from_project_root("Dataset", "splits"))
    parser.add_argument("--results-dir", type=Path, default=from_project_root("results"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--fallback-models", nargs="*", default=["convnext", "convnext_tiny", "resnet50", "efficientnet_b0"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_cli_path(args.dataset_root)
    args.split_dir = resolve_cli_path(args.split_dir)
    args.results_dir = resolve_cli_path(args.results_dir)
    if args.checkpoint is not None:
        args.checkpoint = resolve_cli_path(args.checkpoint)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or args.model.lower()
    args.image_size = resolve_image_size(args.image_size, args.model)
    run_results_dir = Path(args.results_dir) / run_name
    checkpoint_path = args.checkpoint or from_project_root("models", f"best_{run_name}.pth")

    prepared = prepare_dataset(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        results_dir=run_results_dir,
        seed=args.seed,
        force=args.force_split,
    )
    metadata = load_split_metadata(args.split_dir)
    class_names = metadata["class_names"]

    device = get_device()
    data_loaders = create_data_loaders(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        model_name=args.model,
    )
    model, model_info = create_model_with_fallback(
        primary_model=args.model,
        fallback_models=args.fallback_models,
        num_classes=len(class_names),
    )
    model = model.to(device)

    history, best_summary = train_model(
        model=model,
        data_loaders=data_loaders,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        model_name=model_info["model_name"],
        image_size=args.image_size,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        min_learning_rate=args.min_learning_rate,
    )

    history_csv, history_json = save_history(history, run_results_dir, prefix="training_history")
    curves_path = run_results_dir / "training_curves.png"
    plot_training_history(history, curves_path, title_prefix=run_name.replace("_", " ").title())

    best_model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    y_true, y_pred = collect_predictions(best_model, data_loaders["test"], device)
    test_metrics = compute_metrics(y_true, y_pred, class_names)

    metrics_path = run_results_dir / "test_metrics.json"
    confusion_matrix_path = run_results_dir / "confusion_matrix_test.png"
    training_summary_path = run_results_dir / "training_summary.json"

    save_metrics(test_metrics, metrics_path)
    save_confusion_matrix(
        test_metrics,
        class_names,
        confusion_matrix_path,
        title=f"{run_name.replace('_', ' ').title()} Confusion Matrix",
    )

    training_summary = {
        "dataset_analysis": prepared["analysis_json"],
        "split_metadata": prepared["metadata_path"],
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "requested_model": args.model,
        "run_name": run_name,
        "selected_model": model_info["model_name"],
        "selected_pretrained": model_info["pretrained"],
        "fallback_errors": model_info["fallback_errors"],
        "best_epoch": best_summary.get("epoch"),
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
        "epochs_ran": len(history),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "min_learning_rate": args.min_learning_rate,
            "image_size": args.image_size,
        },
        "history_csv": str(history_csv),
        "history_json": str(history_json),
        "training_curves": str(curves_path.resolve()),
        "test_metrics_path": str(metrics_path.resolve()),
        "confusion_matrix_path": str(confusion_matrix_path.resolve()),
        "test_metrics": test_metrics,
    }
    with training_summary_path.open("w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2)

    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
