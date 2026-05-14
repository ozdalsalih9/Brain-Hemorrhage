from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model_utils import _quantize_state_dict_uint8
from project_paths import from_project_root, resolve_cli_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint with uint8-compressed weights.")
    parser.add_argument(
        "--input",
        type=Path,
        default=from_project_root("models", "best_convnext_base_es.pth"),
        help="Path to the source fp32 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=from_project_root("models", "best_convnext_base_es_uint8.pth"),
        help="Path for the compressed checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_cli_path(args.input)
    output_path = resolve_cli_path(args.output)

    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing 'state_dict'.")

    exported = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    exported["quantized_state_dict"] = _quantize_state_dict_uint8(checkpoint["state_dict"])
    exported["compression"] = "uint8_affine_per_tensor"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)

    source_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Exported uint8 checkpoint to {output_path} | "
        f"source={source_size:.2f} MiB | output={output_size:.2f} MiB"
    )


if __name__ == "__main__":
    main()
