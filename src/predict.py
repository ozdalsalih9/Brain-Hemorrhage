from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_utils import get_device, load_checkpoint, predict_image, resolve_image_size
from project_paths import from_project_root, resolve_cli_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the class of a single CT image.")
    parser.add_argument("image_path", type=Path, help="Path to the image file.")
    parser.add_argument("--checkpoint", type=Path, default=from_project_root("models", "best_convnext_base_es_uint8.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.image_path = resolve_cli_path(args.image_path)
    args.checkpoint = resolve_cli_path(args.checkpoint)

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    image_size = resolve_image_size(checkpoint.get("image_size"), checkpoint.get("model_name"))
    prediction = predict_image(
        model=model,
        image_path=args.image_path,
        device=device,
        image_size=image_size,
        class_names=checkpoint["class_names"],
        model_name=checkpoint.get("model_name"),
        normalization_mean=checkpoint.get("normalization_mean"),
        normalization_std=checkpoint.get("normalization_std"),
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
