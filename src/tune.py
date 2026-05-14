from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_split_metadata, prepare_dataset
from model_utils import create_data_loaders, create_model_with_fallback, get_device, resolve_image_size, train_model
from project_paths import from_project_root, resolve_cli_path


def plot_hyperparameter_search(results_csv: Path, output_path: Path) -> None:
    frame = pd.read_csv(results_csv)
    frame = frame.sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)
    labels = [
        f"lr={row.learning_rate}\nbs={row.batch_size}\ne={row.epochs}"
        for row in frame.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, frame["best_val_accuracy"], color="#457b9d")
    ax.set_title("Hyperparameter Search Results")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, value in zip(bars, frame["best_val_accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact hyperparameter tuning for a pretrained CNN.")
    parser.add_argument("--dataset-root", type=Path, default=from_project_root("Dataset"))
    parser.add_argument("--split-dir", type=Path, default=from_project_root("Dataset", "splits"))
    parser.add_argument("--results-dir", type=Path, default=from_project_root("results"))
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[8, 16])
    parser.add_argument("--learning-rates", nargs="*", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--epochs-options", nargs="*", type=int, default=[6])
    parser.add_argument("--weight-decays", nargs="*", type=float, default=[1e-4, 5e-4])
    parser.add_argument("--label-smoothings", nargs="*", type=float, default=[0.05, 0.08])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_cli_path(args.dataset_root)
    args.split_dir = resolve_cli_path(args.split_dir)
    args.results_dir = resolve_cli_path(args.results_dir)
    args.image_size = resolve_image_size(args.image_size, args.model)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tuning_dir = Path(args.results_dir) / "tuning"
    prepare_dataset(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        results_dir=tuning_dir,
        seed=args.seed,
    )
    metadata = load_split_metadata(args.split_dir)
    class_names = metadata["class_names"]
    device = get_device()

    rows = []
    for index, (batch_size, learning_rate, epochs, weight_decay, label_smoothing) in enumerate(
        itertools.product(
            args.batch_sizes,
            args.learning_rates,
            args.epochs_options,
            args.weight_decays,
            args.label_smoothings,
        ),
        start=1,
    ):
        run_name = f"{args.model}_tune_{index}"
        checkpoint_path = from_project_root("models", f"{run_name}.pth")
        data_loaders = create_data_loaders(
            dataset_root=args.dataset_root,
            split_dir=args.split_dir,
            batch_size=batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            model_name=args.model,
        )
        model, model_info = create_model_with_fallback(args.model, ["resnet50", "efficientnet_b0"], len(class_names))
        model = model.to(device)
        history, best_summary = train_model(
            model=model,
            data_loaders=data_loaders,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=args.patience,
            checkpoint_path=checkpoint_path,
            class_names=class_names,
            model_name=model_info["model_name"],
            image_size=args.image_size,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            min_learning_rate=args.min_learning_rate,
        )
        rows.append(
            {
                "run_name": run_name,
                "selected_model": model_info["model_name"],
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "min_learning_rate": args.min_learning_rate,
                "patience": args.patience,
                "best_epoch": best_summary.get("epoch"),
                "best_val_accuracy": best_summary.get("val_accuracy", 0.0),
                "best_val_loss": best_summary.get("val_loss", 0.0),
                "checkpoint": str(checkpoint_path.resolve()),
                "history": json.dumps(history),
            }
        )

    tuning_dir.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_dir / f"{args.model}_tuning_results.csv"
    results_json = tuning_dir / f"{args.model}_tuning_results.json"
    results_plot = tuning_dir / f"{args.model}_tuning_plot.png"

    frame = pd.DataFrame(rows).sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)
    frame.to_csv(results_csv, index=False)
    frame.to_json(results_json, orient="records", indent=2)
    plot_hyperparameter_search(results_csv, results_plot)

    best_row = frame.iloc[0].to_dict()
    best_config_path = tuning_dir / f"{args.model}_best_config.json"
    with best_config_path.open("w", encoding="utf-8") as file:
        json.dump(best_row, file, indent=2)

    print(json.dumps({"results_csv": str(results_csv), "best_config": best_row, "plot": str(results_plot)}, indent=2))


if __name__ == "__main__":
    main()
