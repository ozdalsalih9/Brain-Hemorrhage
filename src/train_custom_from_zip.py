from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model_utils import (
    LegacyFlatCustomCNN,
    collect_predictions,
    compute_metrics,
    get_device,
    load_checkpoint,
    plot_training_history,
    save_checkpoint,
    save_confusion_matrix,
    save_history,
    save_metrics,
)
from project_paths import from_project_root, resolve_cli_path


CLASS_NAMES = ["no_hemorrhage", "hemorrhage"]
EXPERIMENTS = [
    {"lr": 1e-3, "batch_size": 16, "dropout1": 0.5, "dropout2": 0.3, "weight_decay": 1e-4},
    {"lr": 5e-4, "batch_size": 16, "dropout1": 0.5, "dropout2": 0.3, "weight_decay": 1e-4},
    {"lr": 1e-4, "batch_size": 16, "dropout1": 0.5, "dropout2": 0.3, "weight_decay": 1e-4},
    {"lr": 5e-4, "batch_size": 32, "dropout1": 0.5, "dropout2": 0.3, "weight_decay": 1e-4},
    {"lr": 5e-4, "batch_size": 16, "dropout1": 0.3, "dropout2": 0.2, "weight_decay": 1e-4},
    {"lr": 5e-4, "batch_size": 16, "dropout1": 0.5, "dropout2": 0.3, "weight_decay": 0.0},
    {"lr": 1e-4, "batch_size": 32, "dropout1": 0.5, "dropout2": 0.3, "weight_decay": 1e-4},
    {"lr": 5e-4, "batch_size": 32, "dropout1": 0.3, "dropout2": 0.2, "weight_decay": 0.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild and retrain the custom CNN using the zip project's setup.")
    parser.add_argument("--dataset-root", type=Path, default=from_project_root("Dataset"))
    parser.add_argument("--results-dir", type=Path, default=from_project_root("results", "improved_custom_cnn"))
    parser.add_argument("--split-dir", type=Path, default=from_project_root("Dataset", "splits_custom_zip"))
    parser.add_argument("--checkpoint", type=Path, default=from_project_root("models", "custom_cnn.pth"))
    parser.add_argument("--backup-dir", type=Path, default=from_project_root("results", "improved_custom_cnn_backup"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_zip_split(dataset_root: Path, split_dir: Path, seed: int) -> dict[str, Any]:
    labels_path = dataset_root / "labels.csv"
    image_dir = dataset_root / "head_ct" / "head_ct"
    labels = pd.read_csv(labels_path)
    labels.columns = [column.strip().lower() for column in labels.columns]
    labels["filename"] = labels["id"].apply(lambda value: f"{int(value):03d}.png")
    labels["label_idx"] = labels["hemorrhage"].astype(int)
    labels["class_name"] = labels["label_idx"].map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
    labels["image_path"] = labels["filename"].apply(lambda value: str(Path("head_ct") / "head_ct" / value))

    train_df, temp_df = train_test_split(
        labels,
        test_size=0.30,
        stratify=labels["label_idx"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label_idx"],
        random_state=seed,
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    for name, frame in manifests.items():
        output = frame[["id", "filename", "image_path", "label_idx", "class_name"]].sort_values("id")
        output.to_csv(split_dir / f"{name}.csv", index=False)

    metadata = {
        "source": "deep_learning_customCNN.zip split logic",
        "labels_path": str(labels_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "seed": seed,
        "split_strategy": "train_test_split train=70% val=15% test=15% stratified random_state=42",
        "class_names": CLASS_NAMES,
        "counts": {
            name: {
                "total": int(len(frame)),
                "by_class": {class_name: int((frame["class_name"] == class_name).sum()) for class_name in CLASS_NAMES},
            }
            for name, frame in manifests.items()
        },
    }
    with (split_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def get_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class ZipCustomDataset(Dataset):
    def __init__(self, manifest_path: Path, dataset_root: Path, transform: transforms.Compose) -> None:
        self.frame = pd.read_csv(manifest_path).reset_index(drop=True)
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image_path = self.dataset_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(row["label_idx"])


def create_dataloaders(
    dataset_root: Path,
    split_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    datasets = {
        "train": ZipCustomDataset(split_dir / "train.csv", dataset_root, get_train_transform(image_size)),
        "val": ZipCustomDataset(split_dir / "val.csv", dataset_root, get_eval_transform(image_size)),
        "test": ZipCustomDataset(split_dir / "test.csv", dataset_root, get_eval_transform(image_size)),
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return running_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def run_experiment(
    config: dict[str, float | int],
    dataset_root: Path,
    split_dir: Path,
    image_size: int,
    num_workers: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, Any]:
    loaders = create_dataloaders(
        dataset_root=dataset_root,
        split_dir=split_dir,
        image_size=image_size,
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
    )
    model = LegacyFlatCustomCNN(
        dropout1=float(config["dropout1"]),
        dropout2=float(config["dropout2"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history: list[dict[str, Any]] = []
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"event": "zip_custom_epoch", **config, **row}), flush=True)

        improved = val_accuracy > best_val_accuracy or (
            np.isclose(val_accuracy, best_val_accuracy) and val_loss < best_val_loss
        )
        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "event": "zip_custom_early_stopping",
                        **config,
                        "epoch": epoch,
                        "best_val_accuracy": best_val_accuracy,
                    }
                ),
                flush=True,
            )
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    return {
        **config,
        "best_val_acc": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "best_state_dict": best_state,
        "epochs_ran": len(history),
    }


def evaluate_reference_zip_checkpoint(split_dir: Path, dataset_root: Path, device: torch.device) -> dict[str, Any]:
    zip_checkpoint = from_project_root("models", "custom_cnn.pth")
    model, _ = load_checkpoint(zip_checkpoint, device=device)
    loader = create_dataloaders(dataset_root, split_dir, image_size=128, batch_size=16, num_workers=0)["test"]
    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    return {
        "checkpoint": str(zip_checkpoint.resolve()),
        "metrics": metrics,
    }


def backup_existing_results(results_dir: Path, backup_dir: Path) -> None:
    if not results_dir.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in results_dir.iterdir():
        target = backup_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_cli_path(args.dataset_root)
    results_dir = resolve_cli_path(args.results_dir)
    split_dir = resolve_cli_path(args.split_dir)
    checkpoint_path = resolve_cli_path(args.checkpoint)
    backup_dir = resolve_cli_path(args.backup_dir)

    set_seed(args.seed)
    device = get_device()

    results_dir.mkdir(parents=True, exist_ok=True)
    backup_existing_results(results_dir, backup_dir)
    split_metadata = prepare_zip_split(dataset_root, split_dir, args.seed)
    zip_reference = evaluate_reference_zip_checkpoint(split_dir, dataset_root, device)

    experiment_results: list[dict[str, Any]] = []
    for index, config in enumerate(EXPERIMENTS, start=1):
        print(json.dumps({"event": "zip_custom_experiment_start", "index": index, "config": config}), flush=True)
        result = run_experiment(
            config=config,
            dataset_root=dataset_root,
            split_dir=split_dir,
            image_size=args.image_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        experiment_results.append(result)
        print(
            json.dumps(
                {
                    "event": "zip_custom_experiment_end",
                    "index": index,
                    "best_val_acc": result["best_val_acc"],
                    "best_val_loss": result["best_val_loss"],
                }
            ),
            flush=True,
        )

    best_result = max(experiment_results, key=lambda row: (row["best_val_acc"], -row["best_val_loss"]))
    best_model = LegacyFlatCustomCNN(
        dropout1=float(best_result["dropout1"]),
        dropout2=float(best_result["dropout2"]),
    ).to(device)
    best_model.load_state_dict(best_result["best_state_dict"])
    best_model.eval()

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=best_model,
        model_name="improved_custom_cnn_zip",
        class_names=CLASS_NAMES,
        image_size=args.image_size,
        best_val_accuracy=float(best_result["best_val_acc"]),
    )

    test_loader = create_dataloaders(
        dataset_root=dataset_root,
        split_dir=split_dir,
        image_size=args.image_size,
        batch_size=int(best_result["batch_size"]),
        num_workers=args.num_workers,
    )["test"]
    y_true, y_pred = collect_predictions(best_model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)

    histories_for_csv = []
    for idx, result in enumerate(experiment_results, start=1):
        histories_for_csv.append(
            {
                "experiment": idx,
                "lr": result["lr"],
                "batch_size": result["batch_size"],
                "dropout1": result["dropout1"],
                "dropout2": result["dropout2"],
                "weight_decay": result["weight_decay"],
                "best_val_acc": result["best_val_acc"],
                "best_val_loss": result["best_val_loss"],
                "epochs_ran": result["epochs_ran"],
            }
        )
    tuning_csv = results_dir / "zip_tuning_results.csv"
    tuning_json = results_dir / "zip_tuning_results.json"
    pd.DataFrame(histories_for_csv).sort_values(["best_val_acc", "best_val_loss"], ascending=[False, True]).to_csv(
        tuning_csv,
        index=False,
    )
    with tuning_json.open("w", encoding="utf-8") as file:
        json.dump(histories_for_csv, file, indent=2)

    best_history_csv, best_history_json = save_history(best_result["history"], results_dir, prefix="training_history")
    plot_training_history(best_result["history"], results_dir / "training_curves.png", title_prefix="Custom CNN")
    save_metrics(test_metrics, results_dir / "test_metrics.json")
    save_confusion_matrix(
        test_metrics,
        CLASS_NAMES,
        results_dir / "confusion_matrix_test.png",
        title="Confusion Matrix - Custom CNN",
    )

    prediction_frame = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "true_class": [CLASS_NAMES[index] for index in y_true],
            "predicted_class": [CLASS_NAMES[index] for index in y_pred],
        }
    )
    prediction_frame.to_csv(results_dir / "test_predictions.csv", index=False)

    training_summary = {
        "source_basis": "deep_learning_customCNN.zip",
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_model": "improved_custom_cnn_zip",
        "architecture": "zip flat custom CNN (4 conv blocks + flat classifier)",
        "image_size": args.image_size,
        "seed": args.seed,
        "split_dir": str(split_dir.resolve()),
        "split_metadata": split_metadata,
        "zip_reference_checkpoint": zip_reference["checkpoint"],
        "zip_reference_test_metrics": zip_reference["metrics"],
        "hyperparameter_search_space": EXPERIMENTS,
        "best_hyperparameters": {
            "lr": best_result["lr"],
            "batch_size": best_result["batch_size"],
            "dropout1": best_result["dropout1"],
            "dropout2": best_result["dropout2"],
            "weight_decay": best_result["weight_decay"],
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "best_epoch": int(
            max(best_result["history"], key=lambda row: (row["val_accuracy"], -row["val_loss"]))["epoch"]
        ),
        "epochs_ran": best_result["epochs_ran"],
        "history_csv": str(best_history_csv.resolve()),
        "history_json": str(best_history_json.resolve()),
        "tuning_results_csv": str(tuning_csv.resolve()),
        "tuning_results_json": str(tuning_json.resolve()),
        "training_curves": str((results_dir / "training_curves.png").resolve()),
        "test_metrics_path": str((results_dir / "test_metrics.json").resolve()),
        "confusion_matrix_path": str((results_dir / "confusion_matrix_test.png").resolve()),
        "test_metrics": test_metrics,
        "classification_report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
    }
    with (results_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2)

    print(json.dumps(training_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
