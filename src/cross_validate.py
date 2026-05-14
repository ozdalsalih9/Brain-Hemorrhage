from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import analyze_dataset, create_stratified_kfold_manifests, write_analysis_reports
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
    parser = argparse.ArgumentParser(description="Run stratified k-fold cross-validation for hemorrhage classification.")
    parser.add_argument("--dataset-root", type=Path, default=from_project_root("Dataset"))
    parser.add_argument("--results-dir", type=Path, default=from_project_root("results", "cross_validation"))
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--fallback-models", nargs="*", default=["convnext", "convnext_tiny", "resnet50", "efficientnet_b0"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _aggregate_fold_metrics(fold_rows: list[dict]) -> dict[str, float]:
    frame = pd.DataFrame(fold_rows)
    aggregate: dict[str, float] = {}
    for column in ("test_accuracy", "test_precision", "test_recall", "test_f1", "best_val_accuracy", "best_val_loss"):
        aggregate[f"{column}_mean"] = float(frame[column].mean())
        aggregate[f"{column}_std"] = float(frame[column].std(ddof=0))
    aggregate["epochs_ran_mean"] = float(frame["epochs_ran"].mean())
    aggregate["epochs_ran_std"] = float(frame["epochs_ran"].std(ddof=0))
    return aggregate


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_cli_path(args.dataset_root)
    args.results_dir = resolve_cli_path(args.results_dir)
    args.image_size = resolve_image_size(args.image_size, args.model)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"{args.model}_cv"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_dataset(args.dataset_root)
    analysis_json, analysis_txt = write_analysis_reports(analysis, run_dir)
    folds_dir = run_dir / "fold_manifests"
    fold_manifests = create_stratified_kfold_manifests(
        analysis=analysis,
        output_dir=folds_dir,
        folds=args.folds,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed,
    )

    device = get_device()
    class_names = analysis["class_names"]
    fold_rows: list[dict] = []

    for fold_info in fold_manifests:
        fold_index = int(fold_info["fold_index"])
        fold_name = f"fold_{fold_index:02d}"
        fold_dir = run_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = from_project_root("models", f"{run_name}_{fold_name}.pth")

        data_loaders = create_data_loaders(
            dataset_root=args.dataset_root,
            split_dir=fold_info["split_dir"],
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

        history_csv, history_json = save_history(history, fold_dir, prefix="training_history")
        curves_path = fold_dir / "training_curves.png"
        plot_training_history(history, curves_path, title_prefix=f"{run_name} {fold_name}")

        best_model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        y_true, y_pred = collect_predictions(best_model, data_loaders["test"], device)
        metrics = compute_metrics(y_true, y_pred, class_names)

        metrics_path = fold_dir / "test_metrics.json"
        confusion_matrix_path = fold_dir / "confusion_matrix_test.png"
        save_metrics(metrics, metrics_path)
        save_confusion_matrix(
            metrics,
            class_names,
            confusion_matrix_path,
            title=f"{run_name} {fold_name} Confusion Matrix",
        )

        fold_summary = {
            "fold_index": fold_index,
            "split_dir": fold_info["split_dir"],
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "selected_model": model_info["model_name"],
            "selected_pretrained": model_info["pretrained"],
            "best_epoch": best_summary.get("epoch"),
            "best_val_accuracy": float(best_summary.get("val_accuracy", 0.0)),
            "best_val_loss": float(best_summary.get("val_loss", 0.0)),
            "epochs_ran": len(history),
            "test_accuracy": float(metrics["accuracy"]),
            "test_precision": float(metrics["precision"]),
            "test_recall": float(metrics["recall"]),
            "test_f1": float(metrics["f1"]),
            "history_csv": str(history_csv.resolve()),
            "history_json": str(history_json.resolve()),
            "training_curves": str(curves_path.resolve()),
            "test_metrics_path": str(metrics_path.resolve()),
            "confusion_matrix_path": str(confusion_matrix_path.resolve()),
        }
        with (fold_dir / "fold_summary.json").open("w", encoding="utf-8") as file:
            json.dump(fold_summary, file, indent=2)
        fold_rows.append(fold_summary)

    summary = {
        "run_name": run_name,
        "requested_model": args.model,
        "folds": args.folds,
        "dataset_analysis": str(Path(analysis_json).resolve()),
        "dataset_analysis_text": str(Path(analysis_txt).resolve()),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "min_learning_rate": args.min_learning_rate,
            "image_size": args.image_size,
            "val_ratio_within_train": args.val_ratio_within_train,
        },
        "fold_results": fold_rows,
        "aggregate_metrics": _aggregate_fold_metrics(fold_rows),
    }

    fold_results_csv = run_dir / "fold_results.csv"
    pd.DataFrame(fold_rows).to_csv(fold_results_csv, index=False)
    summary["fold_results_csv"] = str(fold_results_csv.resolve())

    summary_path = run_dir / "cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
