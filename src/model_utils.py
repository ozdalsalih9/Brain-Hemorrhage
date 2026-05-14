from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw, ImageOps
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter, label
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CUSTOM_CNN_MEAN = [0.5, 0.5, 0.5]
CUSTOM_CNN_STD = [0.5, 0.5, 0.5]
COMPACT_CUSTOM_MODEL_NAMES = {"improved_custom_cnn", "custom_cnn", "improvedcnn"}
LEGACY_FLAT_CUSTOM_MODEL_NAMES = {"improved_custom_cnn_zip", "improved_custom_cnn_v1", "custom_cnn_v1"}
LEGACY_DEEP_CUSTOM_MODEL_NAMES = {"improved_custom_cnn_legacy", "improved_custom_cnn_deep", "custom_cnn_legacy"}
CUSTOM_MODEL_NAMES = COMPACT_CUSTOM_MODEL_NAMES | LEGACY_FLAT_CUSTOM_MODEL_NAMES | LEGACY_DEEP_CUSTOM_MODEL_NAMES
CONVNEXT_SMALL_NAMES = {"convnext", "convnext_small", "normal_convnext"}
CONVNEXT_BASE_NAMES = {"convnext_base", "convnext_medium", "medium_convnext", "base_convnext"}
INFERENCE_TTA_BLEND = 0.25
INFERENCE_OVERRIDE_CAP = 0.78


class ImprovedCustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout1: float = 0.30, dropout2: float = 0.15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.02),
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
        )
        # Preserve some spatial structure for small hemorrhage cues without
        # returning to the very large fully connected classifier.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LegacyDeepCustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout1: float = 0.50, dropout2: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LegacyFlatCustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout1: float = 0.50, dropout2: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, dataset_root: str | Path, transform=None) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.transform = transform
        self.data = pd.read_csv(self.manifest_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.data.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.dataset_root / image_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, int(row["label_idx"])


class AddGaussianNoise:
    def __init__(self, std_range: tuple[float, float] = (0.0, 0.025), p: float = 0.35) -> None:
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        std = float(torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item())
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_preprocessing(model_name: str | None) -> dict[str, Any]:
    normalized_name = str(model_name or "convnext").lower()
    if normalized_name in LEGACY_DEEP_CUSTOM_MODEL_NAMES:
        return {
            "image_size": 128,
            "mean": CUSTOM_CNN_MEAN,
            "std": CUSTOM_CNN_STD,
        }
    if normalized_name in LEGACY_FLAT_CUSTOM_MODEL_NAMES:
        return {
            "image_size": 128,
            "mean": CUSTOM_CNN_MEAN,
            "std": CUSTOM_CNN_STD,
        }
    if normalized_name in COMPACT_CUSTOM_MODEL_NAMES:
        return {
            "image_size": 160,
            "mean": CUSTOM_CNN_MEAN,
            "std": CUSTOM_CNN_STD,
        }
    return {
        "image_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }


def resolve_image_size(image_size: int | None, model_name: str | None) -> int:
    if image_size is not None:
        return int(image_size)
    return int(get_model_preprocessing(model_name)["image_size"])


def get_train_transforms(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.06)),
            transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.94, 1.06), shear=2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.10),
            transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.15),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.05, scale=(0.01, 0.05)),
            AddGaussianNoise(std_range=(0.0, 0.02), p=0.20),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_eval_transforms(
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    normalized_name = model_name.lower()

    if normalized_name in CONVNEXT_BASE_NAMES:
        try:
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = models.convnext_base(weights=weights)
        except AttributeError:
            model = models.convnext_base(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if normalized_name in CONVNEXT_SMALL_NAMES:
        try:
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = models.convnext_small(weights=weights)
        except AttributeError:
            model = models.convnext_small(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if normalized_name == "convnext_tiny":
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
        except AttributeError:
            model = models.convnext_tiny(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if normalized_name == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if normalized_name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if normalized_name in COMPACT_CUSTOM_MODEL_NAMES:
        return ImprovedCustomCNN(num_classes=num_classes)
    if normalized_name in LEGACY_FLAT_CUSTOM_MODEL_NAMES:
        return LegacyFlatCustomCNN(num_classes=num_classes)
    if normalized_name in LEGACY_DEEP_CUSTOM_MODEL_NAMES:
        return LegacyDeepCustomCNN(num_classes=num_classes)

    raise ValueError(f"Unsupported model name: {model_name}")


def create_model_with_fallback(
    primary_model: str,
    fallback_models: list[str],
    num_classes: int,
) -> tuple[nn.Module, dict[str, Any]]:
    errors: list[str] = []
    candidate_models = [primary_model, *fallback_models]

    for pretrained in (True, False):
        for candidate in candidate_models:
            try:
                model = build_model(candidate, num_classes=num_classes, pretrained=pretrained)
                effective_pretrained = pretrained and candidate.lower() not in CUSTOM_MODEL_NAMES
                return model, {"model_name": candidate, "pretrained": effective_pretrained, "fallback_errors": errors}
            except Exception as exc:
                mode = "pretrained" if pretrained else "random_init"
                errors.append(f"{candidate} ({mode}): {exc}")

    raise RuntimeError("Unable to initialize any model. Details: " + " | ".join(errors))


def create_data_loaders(
    dataset_root: str | Path,
    split_dir: str | Path,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 0,
    model_name: str | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> dict[str, DataLoader]:
    dataset_root = Path(dataset_root).resolve()
    split_dir = Path(split_dir).resolve()
    preprocessing = get_model_preprocessing(model_name)
    mean = mean or preprocessing["mean"]
    std = std or preprocessing["std"]
    datasets = {
        "train": ManifestImageDataset(split_dir / "train.csv", dataset_root, get_train_transforms(image_size, mean, std)),
        "val": ManifestImageDataset(split_dir / "val.csv", dataset_root, get_eval_transforms(image_size, mean, std)),
        "test": ManifestImageDataset(split_dir / "test.csv", dataset_root, get_eval_transforms(image_size, mean, std)),
    }
    pin_memory = torch.cuda.is_available()
    return {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split_name, dataset in datasets.items()
    }


def create_manifest_loader(
    manifest_path: str | Path,
    dataset_root: str | Path,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 0,
    model_name: str | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    shuffle: bool = False,
) -> DataLoader:
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    preprocessing = get_model_preprocessing(model_name)
    mean = mean or preprocessing["mean"]
    std = std or preprocessing["std"]
    dataset = ManifestImageDataset(manifest_path, dataset_root, get_eval_transforms(image_size, mean, std))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    model_name: str,
    class_names: list[str],
    image_size: int,
    best_val_accuracy: float,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing = get_model_preprocessing(model_name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "class_names": class_names,
            "image_size": image_size,
            "best_val_accuracy": best_val_accuracy,
            "normalization_mean": preprocessing["mean"],
            "normalization_std": preprocessing["std"],
        },
        checkpoint_path,
    )


def _quantize_tensor_uint8(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu().to(torch.float32)
    min_value = float(cpu_tensor.min().item())
    max_value = float(cpu_tensor.max().item())
    if max_value <= min_value:
        scale = 1.0
        zero_point = 0
        quantized = torch.zeros_like(cpu_tensor, dtype=torch.uint8)
    else:
        scale = (max_value - min_value) / 255.0
        zero_point = int(round(-min_value / scale))
        zero_point = max(0, min(255, zero_point))
        quantized = torch.clamp(torch.round(cpu_tensor / scale + zero_point), 0, 255).to(torch.uint8)
    return {
        "data": quantized,
        "scale": scale,
        "zero_point": zero_point,
        "shape": list(cpu_tensor.shape),
    }


def _dequantize_tensor_uint8(payload: dict[str, Any]) -> torch.Tensor:
    quantized = payload["data"].to(torch.float32)
    scale = float(payload["scale"])
    zero_point = float(payload["zero_point"])
    restored = (quantized - zero_point) * scale
    return restored.reshape(payload["shape"])


def _quantize_state_dict_uint8(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    quantized_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            quantized_state[key] = {
                "kind": "quantized_uint8",
                **_quantize_tensor_uint8(tensor),
            }
        else:
            quantized_state[key] = {
                "kind": "raw",
                "data": tensor,
            }
    return quantized_state


def _dequantize_state_dict_uint8(quantized_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for key, payload in quantized_state.items():
        kind = payload.get("kind", "raw")
        if kind == "quantized_uint8":
            state_dict[key] = _dequantize_tensor_uint8(payload)
        else:
            state_dict[key] = payload["data"]
    return state_dict


def train_model(
    model: nn.Module,
    data_loaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    checkpoint_path: str | Path,
    class_names: list[str],
    model_name: str,
    image_size: int,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    min_learning_rate: float = 1e-6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.4,
        patience=2,
        min_lr=min_learning_rate,
    )

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    best_summary: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model=model,
            data_loader=data_loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            data_loader=data_loaders["val"],
            criterion=criterion,
            device=device,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    **epoch_summary,
                }
            ),
            flush=True,
        )
        scheduler.step(val_loss)

        improved = val_accuracy > best_val_accuracy or (
            np.isclose(val_accuracy, best_val_accuracy) and val_loss < best_val_loss
        )
        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_summary = epoch_summary.copy()
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                model_name=model_name,
                class_names=class_names,
                image_size=image_size,
                best_val_accuracy=best_val_accuracy,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": epoch,
                        "patience": patience,
                        "best_epoch": best_summary.get("epoch"),
                        "best_val_accuracy": best_val_accuracy,
                    }
                ),
                flush=True,
            )
            break

    return history, best_summary


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "quantized_state_dict" in checkpoint and "state_dict" not in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["state_dict"] = _dequantize_state_dict_uint8(checkpoint["quantized_state_dict"])
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        state_dict = checkpoint if isinstance(checkpoint, dict) else {}
        inferred_model_name = "improved_custom_cnn"
        inferred_image_size = 160
        if "features.21.weight" in state_dict and "features.25.weight" in state_dict:
            inferred_model_name = "improved_custom_cnn_legacy"
            inferred_image_size = 128
        elif "classifier.1.weight" in state_dict and tuple(state_dict["classifier.1.weight"].shape) == (256, 16384):
            inferred_model_name = "improved_custom_cnn_zip"
            inferred_image_size = 128
        checkpoint = {
            "state_dict": checkpoint,
            "model_name": inferred_model_name,
            "class_names": ["no_hemorrhage", "hemorrhage"],
            "image_size": inferred_image_size,
            "best_val_accuracy": None,
            "normalization_mean": CUSTOM_CNN_MEAN,
            "normalization_std": CUSTOM_CNN_STD,
        }

    model_name = str(checkpoint["model_name"]).lower()
    class_names = checkpoint["class_names"]
    preprocessing = get_model_preprocessing(model_name)
    checkpoint.setdefault("image_size", preprocessing["image_size"])
    checkpoint.setdefault("normalization_mean", preprocessing["mean"])
    checkpoint.setdefault("normalization_std", preprocessing["std"])
    model = build_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def plot_training_history(history: list[dict[str, Any]], output_path: str | Path, title_prefix: str = "") -> None:
    frame = pd.DataFrame(history)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train Loss")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss".strip())
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="Train Accuracy")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{title_prefix} Accuracy".strip())
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_history(history: list[dict[str, Any]], results_dir: str | Path, prefix: str = "training_history") -> tuple[Path, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{prefix}.csv"
    json_path = results_dir / f"{prefix}.json"
    pd.DataFrame(history).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    return csv_path, json_path


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    return y_true, y_pred


def collect_prediction_records(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for prediction, confidence, label in zip(
                predictions.cpu().tolist(),
                confidences.cpu().tolist(),
                labels.tolist(),
            ):
                predicted_idx = int(prediction)
                true_idx = int(label)
                records.append(
                    {
                        "predicted_index": predicted_idx,
                        "predicted_class": class_names[predicted_idx],
                        "true_index": true_idx,
                        "true_class": class_names[true_idx],
                        "confidence": float(confidence),
                        "correct": bool(predicted_idx == true_idx),
                    }
                )

    return records


def estimate_prediction_accuracy(
    predicted_class: str,
    confidence: float,
    reference_records: list[dict[str, Any]],
) -> float | None:
    if not reference_records:
        return None

    same_class_records = [record for record in reference_records if record["predicted_class"] == predicted_class]
    candidate_records = same_class_records if len(same_class_records) >= 4 else reference_records
    if not candidate_records:
        return None

    sorted_records = sorted(candidate_records, key=lambda record: abs(record["confidence"] - confidence))
    neighbor_count = min(max(6, len(sorted_records) // 2), len(sorted_records))
    neighbors = sorted_records[:neighbor_count]

    weighted_correct = 0.0
    total_weight = 0.0
    for record in neighbors:
        distance = abs(record["confidence"] - confidence)
        weight = 1.0 / (0.05 + distance)
        weighted_correct += weight * float(record["correct"])
        total_weight += weight

    empirical_accuracy = (weighted_correct + 1.0) / (total_weight + 2.0)
    blended_accuracy = (0.60 * empirical_accuracy) + (0.40 * confidence)
    return float(np.clip(blended_accuracy, 0.0, 1.0))


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred).tolist()
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion,
        "per_class": [
            {
                "class_name": class_name,
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
            for index, class_name in enumerate(class_names)
        ],
    }


def save_confusion_matrix(metrics: dict[str, Any], class_names: list[str], output_path: str | Path, title: str = "Confusion Matrix") -> None:
    matrix = np.array(metrics["confusion_matrix"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device,
    image_size: int,
    class_names: list[str],
    model_name: str | None = None,
    normalization_mean: list[float] | None = None,
    normalization_std: list[float] | None = None,
) -> dict[str, Any]:
    preprocessing = get_model_preprocessing(model_name)
    transform = get_eval_transforms(
        image_size=image_size,
        mean=normalization_mean or preprocessing["mean"],
        std=normalization_std or preprocessing["std"],
    )
    image_path = Path(image_path).resolve()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_idx = int(np.argmax(probabilities))
    return {
        "image_path": str(image_path),
        "predicted_class": class_names[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {
            class_name: float(probabilities[index]) for index, class_name in enumerate(class_names)
        },
    }


def _resolve_target_layer(model: nn.Module) -> nn.Module:
    if isinstance(model, LegacyDeepCustomCNN):
        return model.features[27]
    if isinstance(model, LegacyFlatCustomCNN):
        return model.features[14]
    if isinstance(model, ImprovedCustomCNN):
        # features[17] = last ReLU before final MaxPool+Dropout2d block.
        # With Dropout2d layers the indices shifted from the previous layout.
        return model.features[17]
    if hasattr(model, "features") and len(model.features) > 7:
        # ConvNeXt: features[7] is the last stage (highest-level semantics).
        return model.features[7]
    if hasattr(model, "features") and len(model.features) > 5:
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        return model.features[-1]
    raise ValueError("Grad-CAM target layer could not be inferred for this model.")


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    max_value = float(heatmap.max()) if heatmap.size else 0.0
    if max_value <= 0:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap / max_value).astype(np.float32)


def _estimate_location_text(mask: np.ndarray) -> tuple[str, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape
        center_x = width // 2
        center_y = height // 2
        bbox = (max(center_x - 8, 0), max(center_y - 8, 0), min(center_x + 8, width - 1), min(center_y + 8, height - 1))
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        center_x = int(xs.mean())
        center_y = int(ys.mean())

    height, width = mask.shape
    horizontal = "left" if center_x < width * 0.40 else "right" if center_x > (width * 0.60) else "center"
    vertical = "upper" if center_y < height * 0.40 else "lower" if center_y > (height * 0.60) else "middle"
    if horizontal == "center":
        location = f"{vertical} center"
    else:
        patient_side = "patient right" if horizontal == "left" else "patient left"
        location = f"{vertical} {horizontal} ({patient_side})"
    return location, bbox


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    expansion_ratio: float = 1.35,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1 + 1)
    box_height = max(1, y2 - y1 + 1)
    side = int(max(box_width, box_height) * expansion_ratio)
    side = max(side, 18)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_side = side / 2.0

    new_x1 = int(round(center_x - half_side))
    new_y1 = int(round(center_y - half_side))
    new_x2 = int(round(center_x + half_side))
    new_y2 = int(round(center_y + half_side))

    if new_x1 < 0:
        new_x2 -= new_x1
        new_x1 = 0
    if new_y1 < 0:
        new_y2 -= new_y1
        new_y1 = 0
    if new_x2 >= width:
        shift = new_x2 - width + 1
        new_x1 = max(0, new_x1 - shift)
        new_x2 = width - 1
    if new_y2 >= height:
        shift = new_y2 - height + 1
        new_y1 = max(0, new_y1 - shift)
        new_y2 = height - 1

    return new_x1, new_y1, new_x2, new_y2


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, component_count = label(mask)
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = [(labeled == component_index).sum() for component_index in range(1, component_count + 1)]
    best_index = int(np.argmax(sizes)) + 1
    return labeled == best_index


def _brain_mask_from_image(image: Image.Image) -> np.ndarray:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    threshold = max(0.08, float(np.percentile(grayscale, 45)))
    mask = grayscale > threshold
    mask = _largest_connected_component(mask)
    mask = binary_fill_holes(mask)
    return mask


def _compute_left_right_symmetry(grayscale: np.ndarray, brain_mask: np.ndarray) -> float:
    half_width = grayscale.shape[1] // 2
    if half_width == 0:
        return 1.0

    left_image = grayscale[:, :half_width]
    right_image = np.fliplr(grayscale[:, grayscale.shape[1] - half_width :])
    left_mask = brain_mask[:, :half_width]
    right_mask = np.fliplr(brain_mask[:, brain_mask.shape[1] - half_width :])
    common_mask = left_mask & right_mask
    if not np.any(common_mask):
        return 1.0
    return float(np.abs(left_image[common_mask] - right_image[common_mask]).mean())


def _compute_axis_aligned_edge_ratio(grayscale: np.ndarray) -> tuple[float, float]:
    gradient_y, gradient_x = np.gradient(grayscale)
    magnitude = np.hypot(gradient_x, gradient_y)
    threshold = float(np.percentile(magnitude, 90))
    strong_edges = magnitude >= threshold
    if not np.any(strong_edges):
        return 1.0, 0.0

    axis_aligned = (
        (np.abs(gradient_x) >= 2 * np.abs(gradient_y))
        | (np.abs(gradient_y) >= 2 * np.abs(gradient_x))
    ) & strong_edges
    return float(axis_aligned.sum() / strong_edges.sum()), float(magnitude[strong_edges].mean())


def _compute_skull_ring_features(grayscale: np.ndarray, brain_mask: np.ndarray) -> tuple[float, float, float]:
    if not np.any(brain_mask):
        return 0.0, 0.0, 0.0

    outer_mask = binary_dilation(brain_mask, iterations=4)
    inner_mask = binary_erosion(brain_mask, iterations=6)
    ring_mask = outer_mask & ~inner_mask
    core_mask = binary_erosion(brain_mask, iterations=14)
    if not np.any(core_mask):
        core_mask = inner_mask if np.any(inner_mask) else brain_mask
    if not np.any(ring_mask):
        return 0.0, 0.0, float(grayscale[core_mask].mean()) if np.any(core_mask) else 0.0

    ring_mean = float(grayscale[ring_mask].mean())
    ring_bright_ratio = float((grayscale[ring_mask] > 0.82).mean())
    core_mean = float(grayscale[core_mask].mean()) if np.any(core_mask) else 0.0
    return ring_mean, ring_bright_ratio, core_mean


def inspect_brain_ct(image_path: str | Path) -> dict[str, Any]:
    image_path = Path(image_path).resolve()
    image = Image.open(image_path).convert("L")
    grayscale = np.asarray(image, dtype=np.float32) / 255.0
    height, width = grayscale.shape

    contrast = float(grayscale.std())
    dynamic_range = float(grayscale.max() - grayscale.min())

    brain_mask = _brain_mask_from_image(image.convert("RGB"))
    area_ratio = float(brain_mask.mean())
    ys, xs = np.where(brain_mask)
    if len(xs) == 0 or len(ys) == 0:
        bbox_ratio = 0.0
        fill_ratio = 0.0
        center_offset = 1.0
    else:
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        bbox_ratio = float((bbox_width * bbox_height) / (width * height))
        fill_ratio = float(area_ratio / max(bbox_ratio, 1e-6))
        center_x = float(xs.mean() / width)
        center_y = float(ys.mean() / height)
        center_offset = float(max(abs(center_x - 0.5), abs(center_y - 0.5)))

    symmetry_score = _compute_left_right_symmetry(grayscale, brain_mask)
    border_mask = np.zeros_like(grayscale, dtype=bool)
    border_margin = max(4, int(min(height, width) * 0.06))
    border_mask[:border_margin, :] = True
    border_mask[-border_margin:, :] = True
    border_mask[:, :border_margin] = True
    border_mask[:, -border_margin:] = True
    border_dark_ratio = float((grayscale[border_mask] < 0.12).mean())

    center_region = grayscale[height // 4 : (3 * height) // 4, width // 4 : (3 * width) // 4]
    center_mean = float(center_region.mean())
    center_std = float(center_region.std())
    center_dark_ratio = float((center_region < 0.12).mean())

    axis_edge_ratio, strong_edge_mean = _compute_axis_aligned_edge_ratio(grayscale)
    chart_like = axis_edge_ratio >= 0.92 and strong_edge_mean <= 0.03
    skull_ring_mean, skull_ring_bright_ratio, core_mean = _compute_skull_ring_features(grayscale, brain_mask)
    skull_ring_delta = float(skull_ring_mean - core_mean)
    mri_like = skull_ring_bright_ratio < 0.03 and skull_ring_delta < 0.05 and center_mean > 0.22

    passed_checks = {
        "contrast": contrast >= 0.10,
        "dynamic_range": dynamic_range >= 0.45,
        "area_ratio": 0.30 <= area_ratio <= 0.92,
        "bbox_ratio": bbox_ratio >= 0.40,
        "fill_ratio": fill_ratio >= 0.50,
        "centering": center_offset <= 0.18,
        "symmetry": symmetry_score <= 0.33,
        "skull_ring": skull_ring_bright_ratio >= 0.05 and skull_ring_delta >= 0.06,
    }
    pass_count = sum(1 for passed in passed_checks.values() if passed)
    dark_background_like = border_dark_ratio >= 0.18 and center_mean <= 0.75 and center_dark_ratio <= 0.35
    textured_center_like = 0.14 <= center_mean <= 0.72 and center_std >= 0.04

    anatomy_like = (
        passed_checks["contrast"]
        and passed_checks["dynamic_range"]
        and passed_checks["area_ratio"]
        and passed_checks["bbox_ratio"]
        and passed_checks["fill_ratio"]
        and passed_checks["centering"]
        and passed_checks["symmetry"]
    )
    severe_non_anatomical = (
        pass_count <= 2
        or ((not passed_checks["area_ratio"]) and (not passed_checks["bbox_ratio"]) and center_offset > 0.28)
        or (contrast < 0.05 and dynamic_range < 0.25)
    )

    strong_brain_ct = (
        not chart_like
        and not mri_like
        and passed_checks["contrast"]
        and passed_checks["dynamic_range"]
        and passed_checks["centering"]
        and passed_checks["skull_ring"]
        and (dark_background_like or pass_count >= 7 or (pass_count >= 6 and textured_center_like))
    )
    uncertain_brain_ct = (
        not chart_like and not mri_like and contrast >= 0.08 and dynamic_range >= 0.40 and center_offset <= 0.22
    )
    anatomical_non_ct = not chart_like and mri_like and anatomy_like and pass_count >= 6

    if strong_brain_ct:
        brain_ct_status = "valid"
        brain_ct_message = "Image appears compatible with a brain CT slice."
        brain_ct_should_block = False
    elif anatomical_non_ct:
        brain_ct_status = "uncertain"
        brain_ct_message = (
            "Image appears to be an anatomical brain slice, but the intensity pattern is closer to MRI than CT. "
            "Prediction can continue with caution, although reliability may be lower because the models were trained on CT."
        )
        brain_ct_should_block = False
    elif uncertain_brain_ct or anatomy_like:
        brain_ct_status = "uncertain"
        brain_ct_message = (
            "Image may be a brain CT slice, but framing or intensity pattern is atypical. "
            "Prediction can continue, though reliability may be lower."
        )
        brain_ct_should_block = False
    elif chart_like or severe_non_anatomical:
        brain_ct_status = "invalid"
        reasons: list[str] = []
        if chart_like:
            reasons.append("it looks like a plotted graphic or matrix rather than anatomical imaging")
        if mri_like:
            reasons.append("the image intensity pattern is closer to MRI than CT and lacks a bright cranial bone ring")
        if not passed_checks["contrast"] or not passed_checks["dynamic_range"]:
            reasons.append("grayscale contrast is too weak for a CT slice")
        if not passed_checks["area_ratio"] or not passed_checks["bbox_ratio"] or not passed_checks["fill_ratio"]:
            reasons.append("the main anatomy does not match the expected cranial coverage")
        if not passed_checks["centering"]:
            reasons.append("the dominant structure is not centered like a head CT")
        if not passed_checks["symmetry"]:
            reasons.append("left-right symmetry is weaker than a typical brain slice")
        if not passed_checks["skull_ring"]:
            reasons.append("the bright skull rim expected in non-contrast CT is weak or missing")
        if not reasons:
            reasons.append("the overall intensity pattern does not look like a brain CT")
        brain_ct_message = "Uploaded image does not appear to be a brain CT image because " + "; ".join(reasons) + "."
        brain_ct_should_block = True
    else:
        brain_ct_status = "uncertain"
        brain_ct_message = (
            "Image is not a clean brain CT match, but it is being accepted for comparison because the anatomy appears plausible."
        )
        brain_ct_should_block = False

    is_brain_ct = not brain_ct_should_block

    return {
        "is_brain_ct": is_brain_ct,
        "brain_ct_status": brain_ct_status,
        "brain_ct_message": brain_ct_message,
        "brain_ct_should_block": brain_ct_should_block,
        "brain_ct_score": float(pass_count / len(passed_checks)),
        "brain_area_ratio": area_ratio,
        "brain_bbox_ratio": bbox_ratio,
        "brain_fill_ratio": fill_ratio,
        "brain_center_offset": center_offset,
        "brain_symmetry_score": symmetry_score,
        "brain_border_dark_ratio": border_dark_ratio,
        "brain_center_mean": center_mean,
        "brain_center_std": center_std,
        "brain_axis_edge_ratio": axis_edge_ratio,
        "brain_strong_edge_mean": strong_edge_mean,
        "brain_skull_ring_mean": skull_ring_mean,
        "brain_skull_ring_bright_ratio": skull_ring_bright_ratio,
        "brain_skull_ring_delta": skull_ring_delta,
        "brain_mri_like": mri_like,
        "brain_anatomy_like": anatomy_like,
        "brain_severe_non_anatomical": severe_non_anatomical,
        "brain_anatomical_non_ct": anatomical_non_ct,
        "brain_ct_checks": passed_checks,
    }


def _localization_is_reliable(
    predicted_class: str,
    confidence: float,
    score_map: np.ndarray,
    candidate_mask: np.ndarray,
) -> bool:
    if predicted_class != "hemorrhage":
        return False
    if confidence < 0.55:
        return False

    hot_pixels = int(candidate_mask.sum())
    if hot_pixels == 0:
        return False

    hot_ratio = hot_pixels / float(candidate_mask.size)
    if hot_ratio < 0.0010 or hot_ratio > 0.20:
        return False

    selected_values = score_map[candidate_mask]
    if selected_values.size == 0:
        return False
    if float(selected_values.max()) < 0.30:
        return False
    if float(selected_values.mean()) < 0.12:
        return False
    return True


def _compute_gradcam_pp_map(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    image_size: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    grad_2 = gradient.pow(2)
    grad_3 = grad_2 * gradient
    denominator = (2.0 * grad_2) + (activation * grad_3).sum(dim=(2, 3), keepdim=True)
    denominator = torch.where(
        denominator.abs() > 1e-7,
        denominator,
        torch.full_like(denominator, 1e-7),
    )
    alpha = grad_2 / denominator
    positive_gradient = F.relu(gradient)
    weights = (alpha * positive_gradient).sum(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    heatmap = _normalize_heatmap(cam.squeeze().cpu().numpy())
    heatmap = _normalize_heatmap(heatmap * valid_mask.astype(np.float32))
    return heatmap


def _compute_layercam_map(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    image_size: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    layercam = (F.relu(gradient) * activation).sum(dim=1, keepdim=True)
    layercam = F.relu(layercam)
    layercam = F.interpolate(layercam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    heatmap = _normalize_heatmap(layercam.squeeze().cpu().numpy())
    heatmap = _normalize_heatmap(heatmap * valid_mask.astype(np.float32))
    return heatmap


def _normalize_percentile_map(values: np.ndarray, mask: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(values, dtype=np.float32)
    masked_values = values[mask]
    lower = float(np.percentile(masked_values, low))
    upper = float(np.percentile(masked_values, high))
    if upper <= lower:
        upper = lower + 1e-6
    return np.clip((values - lower) / (upper - lower), 0, 1).astype(np.float32)


def _compute_ct_localization_prior(
    image: Image.Image,
    brain_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    valid_mask = binary_erosion(brain_mask, iterations=6)
    if not np.any(valid_mask):
        valid_mask = binary_erosion(brain_mask, iterations=3)
    if not np.any(valid_mask):
        valid_mask = brain_mask

    local_background = gaussian_filter(grayscale, sigma=5.5)
    bright_residual = np.maximum(grayscale - local_background, 0)

    mirrored = np.fliplr(grayscale)
    mirrored_mask = np.fliplr(valid_mask)
    overlap_mask = valid_mask & mirrored_mask
    asymmetry_raw = np.zeros_like(grayscale, dtype=np.float32)
    asymmetry_raw[overlap_mask] = np.maximum(grayscale[overlap_mask] - mirrored[overlap_mask], 0)

    intensity_map = _normalize_percentile_map(grayscale, valid_mask, 82, 99.7)
    residual_map = _normalize_percentile_map(bright_residual, valid_mask, 80, 99.7)
    asymmetry_map = _normalize_percentile_map(asymmetry_raw, overlap_mask if np.any(overlap_mask) else valid_mask, 75, 99.5)

    prior_map = (0.42 * intensity_map) + (0.38 * residual_map) + (0.20 * asymmetry_map)
    prior_map *= valid_mask.astype(np.float32)
    prior_map = gaussian_filter(prior_map, sigma=0.9)
    return _normalize_heatmap(prior_map), valid_mask, intensity_map, asymmetry_map


def _component_fill_ratio(component_mask: np.ndarray) -> float:
    ys, xs = np.where(component_mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return float(component_mask.sum() / max(bbox_area, 1.0))


def _select_localization_focus(
    score_map: np.ndarray,
    prior_map: np.ndarray,
    heatmap: np.ndarray,
    intensity_map: np.ndarray,
    asymmetry_map: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if not np.any(valid_mask):
        return np.zeros_like(valid_mask, dtype=bool), {
            "peak_score": 0.0,
            "mean_score": 0.0,
            "area_ratio": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "asymmetry_mean": 0.0,
        }

    peak_index = int(np.argmax(score_map * valid_mask.astype(np.float32)))
    peak_y, peak_x = np.unravel_index(peak_index, score_map.shape)
    peak_score = float(score_map[peak_y, peak_x])
    threshold = max(
        0.44,
        min(0.86, peak_score * 0.68),
        float(np.percentile(score_map[valid_mask], 97.5)),
    )
    candidate_mask = (score_map >= threshold) & valid_mask
    labeled, component_count = label(candidate_mask)
    if component_count == 0:
        return np.zeros_like(valid_mask, dtype=bool), {
            "peak_score": peak_score,
            "mean_score": 0.0,
            "area_ratio": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "asymmetry_mean": 0.0,
        }

    best_mask = np.zeros_like(candidate_mask, dtype=bool)
    best_score = -1.0
    best_stats = {
        "peak_score": peak_score,
        "mean_score": 0.0,
        "area_ratio": 0.0,
        "fill_ratio": 0.0,
        "aspect_ratio": 0.0,
        "asymmetry_mean": 0.0,
    }
    valid_area = float(max(valid_mask.sum(), 1))

    for component_index in range(1, component_count + 1):
        component_mask = labeled == component_index
        area = int(component_mask.sum())
        if area < max(12, int(valid_area * 0.0010)):
            continue
        ys, xs = np.where(component_mask)
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect_ratio = max(width, height) / max(min(width, height), 1.0)
        area_ratio = float(area / valid_area)
        fill_ratio = _component_fill_ratio(component_mask)
        peak_component_score = float(score_map[component_mask].max())
        mean_score = float(score_map[component_mask].mean())
        prior_mean = float(prior_map[component_mask].mean())
        heatmap_mean = float(heatmap[component_mask].mean())
        intensity_mean = float(intensity_map[component_mask].mean())
        asymmetry_mean = float(asymmetry_map[component_mask].mean())
        contains_peak = bool(component_mask[peak_y, peak_x])

        component_score = (
            0.34 * peak_component_score
            + 0.22 * mean_score
            + 0.18 * prior_mean
            + 0.14 * heatmap_mean
            + 0.07 * intensity_mean
            + 0.05 * asymmetry_mean
        )
        if contains_peak:
            component_score *= 1.35
        if area_ratio > 0.16:
            component_score *= 0.55
        elif area_ratio > 0.10:
            component_score *= 0.75
        if fill_ratio < 0.14:
            component_score *= 0.60
        elif fill_ratio < 0.20:
            component_score *= 0.82
        if aspect_ratio > 2.5:
            component_score *= 0.65
        elif aspect_ratio > 1.9:
            component_score *= 0.82
        if asymmetry_mean < 0.08 and area_ratio < 0.010:
            component_score *= 0.78

        if component_score > best_score:
            best_score = component_score
            best_mask = component_mask
            best_stats = {
                "peak_score": peak_component_score,
                "mean_score": mean_score,
                "area_ratio": area_ratio,
                "fill_ratio": fill_ratio,
                "aspect_ratio": aspect_ratio,
                "asymmetry_mean": asymmetry_mean,
            }

    return best_mask, best_stats


def _soft_peak_focus_mask(
    score_map: np.ndarray,
    prior_map: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if not np.any(valid_mask):
        return np.zeros_like(valid_mask, dtype=bool), {
            "peak_score": 0.0,
            "mean_score": 0.0,
            "area_ratio": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "asymmetry_mean": 0.0,
        }

    focus_map = _normalize_heatmap((0.72 * score_map) + (0.28 * prior_map))
    peak_index = int(np.argmax(focus_map * valid_mask.astype(np.float32)))
    peak_y, peak_x = np.unravel_index(peak_index, focus_map.shape)
    peak_score = float(focus_map[peak_y, peak_x])
    threshold = max(0.36, min(0.78, peak_score * 0.58))
    mask = (focus_map >= threshold) & valid_mask
    labeled, component_count = label(mask)
    if component_count == 0:
        return np.zeros_like(valid_mask, dtype=bool), {
            "peak_score": peak_score,
            "mean_score": 0.0,
            "area_ratio": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "asymmetry_mean": 0.0,
        }

    peak_label = int(labeled[peak_y, peak_x])
    if peak_label == 0:
        return np.zeros_like(valid_mask, dtype=bool), {
            "peak_score": peak_score,
            "mean_score": 0.0,
            "area_ratio": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "asymmetry_mean": 0.0,
        }

    component_mask = labeled == peak_label
    component_mask = binary_dilation(component_mask, iterations=1) & valid_mask
    ys, xs = np.where(component_mask)
    width = float(xs.max() - xs.min() + 1) if len(xs) else 1.0
    height = float(ys.max() - ys.min() + 1) if len(ys) else 1.0
    aspect_ratio = max(width, height) / max(min(width, height), 1.0)
    area_ratio = float(component_mask.sum() / max(valid_mask.sum(), 1))
    fill_ratio = _component_fill_ratio(component_mask)
    mean_score = float(focus_map[component_mask].mean()) if np.any(component_mask) else 0.0

    stats = {
        "peak_score": peak_score,
        "mean_score": mean_score,
        "area_ratio": area_ratio,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "asymmetry_mean": 0.0,
    }
    return component_mask, stats


def _build_inverted_variant(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.autocontrast(image.convert("L"))
    return ImageOps.invert(grayscale).convert("RGB")


def _compute_candidate_map(
    image: Image.Image,
    brain_mask: np.ndarray,
    allow_dark: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    inner_mask = binary_erosion(brain_mask, iterations=10)
    if not np.any(inner_mask):
        inner_mask = brain_mask

    local_background = gaussian_filter(grayscale, sigma=4)
    bright_contrast = np.maximum(grayscale - local_background, 0)
    dark_contrast = np.maximum(local_background - grayscale, 0)

    masked_values = grayscale[inner_mask]
    bright_contrast_values = bright_contrast[inner_mask]

    p70 = float(np.percentile(masked_values, 70))
    p995 = float(np.percentile(masked_values, 99.5))
    c75 = float(np.percentile(bright_contrast_values, 75))
    c995 = float(np.percentile(bright_contrast_values, 99.5))

    intensity_score = np.clip((grayscale - p70) / max(p995 - p70, 1e-6), 0, 1)
    contrast_score = np.clip((bright_contrast - c75) / max(c995 - c75, 1e-6), 0, 1)
    bright_candidate_map = (0.6 * intensity_score + 0.4 * contrast_score) * inner_mask.astype(np.float32)

    dark_candidate_map = np.zeros_like(bright_candidate_map, dtype=np.float32)
    candidate_map = bright_candidate_map
    if allow_dark:
        p30 = float(np.percentile(masked_values, 30))
        p005 = float(np.percentile(masked_values, 0.5))
        dark_values = dark_contrast[inner_mask]
        d75 = float(np.percentile(dark_values, 75))
        d995 = float(np.percentile(dark_values, 99.5))
        dark_intensity_score = np.clip((p30 - grayscale) / max(p30 - p005, 1e-6), 0, 1)
        dark_contrast_score = np.clip((dark_contrast - d75) / max(d995 - d75, 1e-6), 0, 1)
        dark_candidate_map = (0.58 * dark_intensity_score + 0.42 * dark_contrast_score) * inner_mask.astype(np.float32)
        candidate_map = np.maximum(bright_candidate_map, 0.95 * dark_candidate_map)

    candidate_map = gaussian_filter(candidate_map, sigma=1.2)
    bright_candidate_map = gaussian_filter(bright_candidate_map, sigma=1.0)
    dark_candidate_map = gaussian_filter(dark_candidate_map, sigma=1.0)

    return (
        _normalize_heatmap(candidate_map),
        inner_mask,
        _normalize_heatmap(bright_candidate_map),
        _normalize_heatmap(dark_candidate_map),
    )


def _component_mode_penalty(
    component_mask: np.ndarray,
    bright_map: np.ndarray,
    dark_map: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[float, str]:
    ys, xs = np.where(component_mask)
    if len(xs) == 0 or len(ys) == 0:
        return 1.0, "unknown"

    height, width = component_mask.shape
    box_width = float(xs.max() - xs.min() + 1)
    box_height = float(ys.max() - ys.min() + 1)
    aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
    center_x = float(xs.mean() / width)
    center_distance = abs(center_x - 0.5)

    bright_mean = float(bright_map[component_mask].mean()) if bright_map.size else 0.0
    dark_mean = float(dark_map[component_mask].mean()) if dark_map.size else 0.0
    heatmap_mean = float(heatmap[component_mask].mean()) if heatmap.size else 0.0
    mode = "dark" if dark_mean > (bright_mean * 1.08) else "bright"

    penalty = 1.0
    if mode == "dark":
        if center_distance < 0.11:
            penalty *= 0.28
        elif center_distance < 0.17:
            penalty *= 0.55
        if aspect_ratio > 2.2:
            penalty *= 0.45
        elif aspect_ratio > 1.7:
            penalty *= 0.72
        if heatmap_mean < 0.16:
            penalty *= 0.35
        elif heatmap_mean < 0.22:
            penalty *= 0.65

    return penalty, mode


def _component_stats(
    component_mask: np.ndarray,
    bright_map: np.ndarray,
    dark_map: np.ndarray,
    heatmap: np.ndarray,
    score_map: np.ndarray | None = None,
) -> dict[str, float | str]:
    ys, xs = np.where(component_mask)
    if len(xs) == 0 or len(ys) == 0:
        return {
            "mode": "unknown",
            "aspect_ratio": 1.0,
            "center_distance": 0.0,
            "bright_mean": 0.0,
            "dark_mean": 0.0,
            "heatmap_mean": 0.0,
            "area_ratio": 0.0,
        }

    height, width = component_mask.shape
    box_width = float(xs.max() - xs.min() + 1)
    box_height = float(ys.max() - ys.min() + 1)
    aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
    center_x = float(xs.mean() / width)
    center_distance = abs(center_x - 0.5)
    bright_mean = float(bright_map[component_mask].mean()) if bright_map.size else 0.0
    dark_mean = float(dark_map[component_mask].mean()) if dark_map.size else 0.0
    heatmap_mean = float(heatmap[component_mask].mean()) if heatmap.size else 0.0
    score_mean = float(score_map[component_mask].mean()) if score_map is not None and score_map.size else 0.0
    peak_score = float(score_map[component_mask].max()) if score_map is not None and score_map.size else 0.0
    mode = "dark" if dark_mean > (bright_mean * 1.08) else "bright"
    area_ratio = float(component_mask.sum() / component_mask.size)
    bbox_area = float(box_width * box_height)
    fill_ratio = float(component_mask.sum() / max(bbox_area, 1.0))
    return {
        "mode": mode,
        "aspect_ratio": aspect_ratio,
        "center_distance": center_distance,
        "bright_mean": bright_mean,
        "dark_mean": dark_mean,
        "heatmap_mean": heatmap_mean,
        "score_mean": score_mean,
        "peak_score": peak_score,
        "area_ratio": area_ratio,
        "fill_ratio": fill_ratio,
    }


def _pick_best_component(
    score_map: np.ndarray,
    candidate_map: np.ndarray,
    heatmap: np.ndarray,
    valid_mask: np.ndarray,
    bright_candidate_map: np.ndarray | None = None,
    dark_candidate_map: np.ndarray | None = None,
) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros_like(valid_mask, dtype=bool)

    peak_index = int(np.argmax(score_map * valid_mask.astype(np.float32)))
    peak_y, peak_x = np.unravel_index(peak_index, score_map.shape)
    peak_score = float(score_map[peak_y, peak_x])
    threshold = max(
        0.42,
        min(0.82, peak_score * 0.62),
        float(np.percentile(score_map[valid_mask], 96)) if np.any(valid_mask) else 0.42,
    )
    binary_candidates = score_map >= threshold
    if bright_candidate_map is not None:
        bright_threshold = max(
            0.22,
            float(np.percentile(bright_candidate_map[valid_mask], 82)) if np.any(valid_mask) else 0.22,
        )
        binary_candidates &= bright_candidate_map >= bright_threshold
    binary_candidates &= valid_mask
    labeled, component_count = label(binary_candidates)
    if component_count == 0:
        return np.zeros_like(binary_candidates, dtype=bool)

    component_entries: list[dict[str, Any]] = []
    strongest_bright_score = 0.0
    best_mask = np.zeros_like(binary_candidates, dtype=bool)
    best_score = -1.0
    minimum_area = max(18, int(valid_mask.sum() * 0.0015))
    for component_index in range(1, component_count + 1):
        component_mask = labeled == component_index
        area = int(component_mask.sum())
        if area < minimum_area:
            continue
        score_mean = float(score_map[component_mask].mean())
        candidate_mean = float(candidate_map[component_mask].mean())
        heatmap_mean = float(heatmap[component_mask].mean())
        area_weight = float(np.sqrt(area))
        mode_penalty, mode = _component_mode_penalty(
            component_mask,
            bright_candidate_map if bright_candidate_map is not None else candidate_map,
            dark_candidate_map if dark_candidate_map is not None else np.zeros_like(candidate_map),
            heatmap,
        )
        component_score = (
            0.42 * score_mean
            + 0.18 * candidate_mean
            + 0.30 * heatmap_mean
            + 0.10 * float(score_map[component_mask].max())
        )
        component_score *= min(area_weight, 9.0) * mode_penalty

        stats = _component_stats(
            component_mask,
            bright_candidate_map if bright_candidate_map is not None else candidate_map,
            dark_candidate_map if dark_candidate_map is not None else np.zeros_like(candidate_map),
            heatmap,
            score_map=score_map,
        )
        contains_peak = bool(component_mask[peak_y, peak_x])
        if contains_peak:
            component_score *= 1.55
        if float(stats["fill_ratio"]) < 0.18:
            component_score *= 0.70
        if float(stats["aspect_ratio"]) > 2.1:
            component_score *= 0.72
        if float(stats["area_ratio"]) > 0.075:
            component_score *= 0.58
        elif float(stats["area_ratio"]) > 0.050:
            component_score *= 0.80
        if float(stats["peak_score"]) < 0.58:
            component_score *= 0.70
        component_entries.append(
            {
                "mask": component_mask,
                "score": component_score,
                "mode": mode,
                "stats": stats,
                "contains_peak": contains_peak,
            }
        )
        if mode == "bright":
            strongest_bright_score = max(strongest_bright_score, component_score)

    for entry in component_entries:
        adjusted_score = float(entry["score"])
        stats = entry["stats"]
        if (
            entry["mode"] == "dark"
            and strongest_bright_score >= 0.78 * entry["score"]
        ):
            adjusted_score *= 0.20
        if (
            entry["mode"] == "dark"
            and float(stats["center_distance"]) < 0.14
            and float(stats["aspect_ratio"]) > 1.45
        ):
            adjusted_score *= 0.18
        if not entry["contains_peak"]:
            adjusted_score *= 0.82
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_mask = entry["mask"]

    return best_mask


def _extract_soft_focus_mask(
    score_map: np.ndarray,
    heatmap: np.ndarray,
    candidate_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros_like(valid_mask, dtype=bool)

    focus_map = _normalize_heatmap((0.62 * heatmap) + (0.38 * candidate_map))
    focus_map *= valid_mask.astype(np.float32)
    if float(focus_map.max()) <= 0:
        return np.zeros_like(valid_mask, dtype=bool)

    peak_index = int(np.argmax(focus_map))
    peak_y, peak_x = np.unravel_index(peak_index, focus_map.shape)
    peak_value = float(focus_map[peak_y, peak_x])

    threshold = max(0.34, min(0.78, peak_value * 0.58))
    mask = focus_map >= threshold
    mask &= valid_mask
    labeled, component_count = label(mask)
    if component_count == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    peak_label = int(labeled[peak_y, peak_x])
    if peak_label == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    component_mask = labeled == peak_label
    component_mask = binary_dilation(component_mask, iterations=1)
    component_mask &= valid_mask

    stats = _component_stats(component_mask, candidate_map, np.zeros_like(candidate_map), heatmap, score_map=focus_map)
    if float(stats["area_ratio"]) > 0.065:
        tighter_threshold = max(0.42, min(0.86, peak_value * 0.70))
        tighter_mask = focus_map >= tighter_threshold
        tighter_mask &= valid_mask
        labeled_tight, tight_count = label(tighter_mask)
        if tight_count > 0:
            tight_label = int(labeled_tight[peak_y, peak_x])
            if tight_label != 0:
                component_mask = labeled_tight == tight_label

    stats = _component_stats(component_mask, candidate_map, np.zeros_like(candidate_map), heatmap, score_map=focus_map)
    if (
        float(stats["area_ratio"]) < 0.0012
        or float(stats["area_ratio"]) > 0.070
        or float(stats["aspect_ratio"]) > 2.4
        or float(stats["fill_ratio"]) < 0.16
        or float(stats["peak_score"]) < 0.52
    ):
        return np.zeros_like(valid_mask, dtype=bool)

    return component_mask


def _create_overlay_images(
    original_image: Image.Image,
    heatmap: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[Image.Image, Image.Image]:
    colormap = plt.get_cmap("jet")
    colored = np.uint8(colormap(heatmap)[..., :3] * 255)
    heatmap_image = Image.fromarray(colored).resize(original_image.size, Image.Resampling.BILINEAR)

    overlay = Image.blend(original_image.convert("RGB"), heatmap_image, alpha=0.30)
    if bbox is not None:
        draw = ImageDraw.Draw(overlay)
        scale_x = original_image.width / heatmap.shape[1]
        scale_y = original_image.height / heatmap.shape[0]
        x1, y1, x2, y2 = _expand_bbox(bbox, heatmap.shape)
        rectangle = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int((x2 + 1) * scale_x),
            int((y2 + 1) * scale_y),
        )
        draw.rectangle(rectangle, outline=ImageColor.getrgb("#ff6b6b"), width=5)
    return overlay, heatmap_image


def explain_prediction(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device,
    image_size: int,
    class_names: list[str],
    model_name: str | None = None,
    normalization_mean: list[float] | None = None,
    normalization_std: list[float] | None = None,
    scan_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    normalized_model_name = str(model_name or "").lower()
    allow_inverted_view = bool(
        scan_info is not None
        and (
            scan_info.get("brain_mri_like")
            or scan_info.get("brain_anatomical_non_ct")
        )
    )
    image_path = Path(image_path).resolve()
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    brain_mask = _brain_mask_from_image(resized)
    localization_prior_map, localization_valid_mask, intensity_prior_map, asymmetry_prior_map = _compute_ct_localization_prior(
        resized,
        brain_mask,
    )
    candidate_map, inner_mask, bright_candidate_map, dark_candidate_map = _compute_candidate_map(
        resized,
        brain_mask,
        allow_dark=allow_inverted_view,
    )
    preprocessing = get_model_preprocessing(model_name)
    transform = get_eval_transforms(
        image_size=image_size,
        mean=normalization_mean or preprocessing["mean"],
        std=normalization_std or preprocessing["std"],
    )
    tensor = transform(original).unsqueeze(0).to(device)
    selected_variant = "original"
    selected_tensor = tensor
    selected_candidate_map = candidate_map
    selected_bright_candidate_map = bright_candidate_map
    selected_dark_candidate_map = dark_candidate_map

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = _resolve_target_layer(model)

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        def _compute_augmented_logits(input_tensor: torch.Tensor) -> torch.Tensor:
            logits_local = model(input_tensor)
            with torch.no_grad():
                flipped_tensor = torch.flip(input_tensor, dims=[3])
                flipped_logits = model(flipped_tensor)
            return ((1.0 - INFERENCE_TTA_BLEND) * logits_local) + (INFERENCE_TTA_BLEND * flipped_logits)

        with torch.no_grad():
            selected_logits = _compute_augmented_logits(tensor)
            if allow_inverted_view:
                inverted_original = _build_inverted_variant(original)
                inverted_resized = inverted_original.resize((image_size, image_size), Image.Resampling.BILINEAR)
                inverted_candidate_map, _, inverted_bright_map, inverted_dark_map = _compute_candidate_map(
                    inverted_resized,
                    brain_mask,
                    allow_dark=True,
                )
                inverted_tensor = transform(inverted_original).unsqueeze(0).to(device)
                inverted_logits = _compute_augmented_logits(inverted_tensor)
                original_probs = torch.softmax(selected_logits, dim=1).squeeze(0)
                inverted_probs = torch.softmax(inverted_logits, dim=1).squeeze(0)
                if "hemorrhage" in class_names:
                    hemorrhage_index = class_names.index("hemorrhage")
                    original_hemo = float(original_probs[hemorrhage_index].item())
                    inverted_hemo = float(inverted_probs[hemorrhage_index].item())
                    choose_inverted = (
                        original_hemo < 0.58
                        and (
                            inverted_hemo >= original_hemo + 0.12
                            or (original_hemo < 0.45 and inverted_hemo >= 0.56)
                            or (original_hemo < 0.55 and inverted_hemo >= 0.62)
                        )
                    )
                    if choose_inverted:
                        selected_variant = "inverted"
                        selected_tensor = inverted_tensor
                        selected_candidate_map = inverted_candidate_map
                        selected_bright_candidate_map = inverted_bright_map
                        selected_dark_candidate_map = inverted_dark_map
                        selected_logits = inverted_logits

        probabilities = torch.softmax(selected_logits, dim=1).squeeze(0)
        predicted_idx = int(probabilities.argmax().item())

        activations.clear()
        gradients.clear()
        model.zero_grad(set_to_none=True)
        logits = model(selected_tensor)
        target_logit = logits[:, predicted_idx]
        target_logit.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")

        activation = activations[0]
        gradient = gradients[0]
        gradcam_pp_map = _compute_gradcam_pp_map(activation, gradient, image_size, inner_mask)
        layercam_map = _compute_layercam_map(activation, gradient, image_size, inner_mask)
        heatmap = _normalize_heatmap((0.68 * gradcam_pp_map) + (0.32 * layercam_map))

        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx].item())
        score_map = _normalize_heatmap((0.40 * localization_prior_map) + (0.45 * heatmap) + (0.15 * selected_candidate_map))
        focus_mask, focus_stats = _select_localization_focus(
            score_map=score_map,
            prior_map=localization_prior_map,
            heatmap=heatmap,
            intensity_map=intensity_prior_map,
            asymmetry_map=asymmetry_prior_map,
            valid_mask=localization_valid_mask,
        )
        soft_focus_mask, soft_focus_stats = _soft_peak_focus_mask(
            score_map=score_map,
            prior_map=localization_prior_map,
            valid_mask=localization_valid_mask,
        )
        reliable_localization = _localization_is_reliable(predicted_class, confidence, score_map, focus_mask)
        tentative_localization = (
            predicted_class == "hemorrhage"
            and not reliable_localization
            and np.any(soft_focus_mask)
            and confidence >= 0.78
            and float(soft_focus_stats["peak_score"]) >= 0.58
            and float(soft_focus_stats["mean_score"]) >= 0.24
            and float(soft_focus_stats["area_ratio"]) <= 0.08
            and float(soft_focus_stats["fill_ratio"]) >= 0.14
        )

        if (
            predicted_class != "hemorrhage"
            and allow_inverted_view
            and np.any(focus_mask)
            and "hemorrhage" in class_names
        ):
            focus_mask = _largest_connected_component(focus_mask)
            area_ratio = float(focus_mask.sum() / max(localization_valid_mask.sum(), 1))
            peak_score = float(score_map[focus_mask].max()) if np.any(focus_mask) else 0.0
            mean_score = float(score_map[focus_mask].mean()) if np.any(focus_mask) else 0.0
            candidate_peak = float(localization_prior_map[focus_mask].max()) if np.any(focus_mask) else 0.0
            mode_penalty, component_mode = _component_mode_penalty(
                focus_mask,
                selected_bright_candidate_map,
                selected_dark_candidate_map,
                heatmap,
            )
            ys, xs = np.where(focus_mask)
            box_width = float(xs.max() - xs.min() + 1) if len(xs) else 1.0
            box_height = float(ys.max() - ys.min() + 1) if len(ys) else 1.0
            aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
            center_x = float(xs.mean() / focus_mask.shape[1]) if len(xs) else 0.5
            center_distance = abs(center_x - 0.5)

            allow_override = (
                0.002 <= area_ratio <= 0.20
                and peak_score >= 0.60
                and candidate_peak >= 0.60
            )
            if component_mode == "dark":
                allow_override = (
                    allow_override
                    and area_ratio <= 0.12
                    and peak_score >= 0.72
                    and candidate_peak >= 0.72
                    and mean_score >= 0.20
                    and mode_penalty >= 0.72
                    and center_distance >= 0.12
                    and aspect_ratio <= 1.8
                )

            if allow_override:
                hemorrhage_index = class_names.index("hemorrhage")
                override_probability = max(
                    float(probabilities[hemorrhage_index].item()),
                    min(INFERENCE_OVERRIDE_CAP, 0.18 + (0.42 * peak_score) + (0.24 * candidate_peak) + (0.16 * mean_score)),
                )
                if override_probability >= 0.56:
                    probabilities = probabilities.clone()
                    predicted_idx = hemorrhage_index
                    predicted_class = "hemorrhage"
                    confidence = float(override_probability)
                    probabilities[hemorrhage_index] = confidence
                    if len(class_names) == 2 and "no_hemorrhage" in class_names:
                        probabilities[class_names.index("no_hemorrhage")] = float(1.0 - confidence)

                    location_text = "localization unavailable"
                    bbox = None
                    blank_heatmap = np.zeros_like(selected_candidate_map, dtype=np.float32)
                    overlay_image, heatmap_image = _create_overlay_images(resized, blank_heatmap, None)
                    model_label = "Custom CNN" if normalized_model_name in CUSTOM_MODEL_NAMES else "Pretrained CNN"
                    explanation = (
                        f"{model_label} detected a strong focal anomaly on an atypical accepted brain image and flags hemorrhage for review, but localization is withheld because it is not reliable."
                    )
                    return {
                        "image_path": str(image_path),
                        "predicted_class": predicted_class,
                        "predicted_index": predicted_idx,
                        "confidence": confidence,
                        "localization_available": False,
                        "localization_status": "unavailable",
                        "probabilities": {
                            class_name: float(probabilities[index].item()) for index, class_name in enumerate(class_names)
                        },
                        "location_text": location_text,
                        "bbox": bbox,
                        "explanation_text": explanation,
                        "overlay_image": overlay_image,
                        "heatmap_image": heatmap_image,
                        "used_inverted_view": selected_variant == "inverted",
                    }

        if reliable_localization:
            focus_mask = _largest_connected_component(focus_mask)
            location_text, bbox = _estimate_location_text(focus_mask)
            localized_map = _normalize_heatmap(score_map * focus_mask.astype(np.float32))
            localized_map = gaussian_filter(localized_map, sigma=0.55)
            localized_map = _normalize_heatmap(localized_map)
            overlay_image, heatmap_image = _create_overlay_images(resized, localized_map, bbox)
            explanation = f"Approximate suspicious focus is in the {location_text}."
            localization_status = "reliable"
        elif tentative_localization:
            focus_mask = _largest_connected_component(soft_focus_mask)
            location_text, bbox = _estimate_location_text(focus_mask)
            tentative_map = _normalize_heatmap(score_map * focus_mask.astype(np.float32))
            tentative_map = gaussian_filter(tentative_map, sigma=0.45)
            tentative_map = _normalize_heatmap(tentative_map)
            overlay_image, heatmap_image = _create_overlay_images(resized, tentative_map, bbox)
            explanation = (
                f"Tentative suspicious focus is in the {location_text}; localization is softer and should be reviewed cautiously."
            )
            localization_status = "tentative"
        elif predicted_class == "hemorrhage":
            location_text = "localization unavailable"
            bbox = None
            blank_heatmap = np.zeros_like(heatmap, dtype=np.float32)
            overlay_image, heatmap_image = _create_overlay_images(resized, blank_heatmap, None)
            explanation = (
                "Hemorrhage class was predicted, but focal localization is not reliable enough to display."
            )
            localization_status = "unavailable"
        else:
            location_text = "no focal region shown"
            bbox = None
            blank_heatmap = np.zeros_like(heatmap, dtype=np.float32)
            overlay_image, heatmap_image = _create_overlay_images(resized, blank_heatmap, None)
            explanation = "No definite hemorrhage was predicted, so no suspicious region is displayed."
            localization_status = "negative"

        return {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": confidence,
            "localization_available": reliable_localization,
            "localization_status": localization_status,
            "probabilities": {
                class_name: float(probabilities[index].item()) for index, class_name in enumerate(class_names)
            },
            "location_text": location_text,
            "bbox": bbox,
            "explanation_text": explanation,
            "overlay_image": overlay_image,
            "heatmap_image": heatmap_image,
            "used_inverted_view": selected_variant == "inverted",
        }
    finally:
        forward_handle.remove()
        backward_handle.remove()


def analyze_scan(image_path: str | Path) -> dict[str, Any]:
    image_path = Path(image_path).resolve()
    image = Image.open(image_path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    normalized = array / 255.0
    height, width = normalized.shape

    bright_ratio = float((normalized > 0.78).mean())
    dark_ratio = float((normalized < 0.08).mean())
    mean_intensity = float(normalized.mean())
    contrast = float(normalized.std())
    dynamic_range = float(normalized.max() - normalized.min())

    if contrast > 0.22:
        quality = "high contrast"
    elif contrast > 0.14:
        quality = "medium contrast"
    else:
        quality = "low contrast"

    if bright_ratio > 0.12:
        density_pattern = "contains broad hyperdense regions"
    elif bright_ratio > 0.05:
        density_pattern = "contains moderate hyperdense signal"
    else:
        density_pattern = "does not show strong hyperdense dominance"

    if dark_ratio > 0.28:
        framing = "slice includes wide background margins"
    elif dark_ratio > 0.18:
        framing = "slice is moderately centered"
    else:
        framing = "slice is tightly centered on the cranial region"

    return {
        "width": int(width),
        "height": int(height),
        "mean_intensity": mean_intensity,
        "contrast": contrast,
        "dynamic_range": dynamic_range,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "quality": quality,
        "density_pattern": density_pattern,
        "framing": framing,
        **inspect_brain_ct(image_path),
    }
