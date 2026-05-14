from __future__ import annotations

import argparse
from pathlib import Path

import torch

from project_paths import from_project_root, resolve_cli_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint with fp16 weights to reduce file size.")
    parser.add_argument(
        "--input",
        type=Path,
        default=from_project_root("models", "best_convnext_tiny_ct.pth"),
        help="Path to the source checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=from_project_root("models", "best_convnext_tiny_ct_fp16.pth"),
        help="Path for the reduced-size checkpoint.",
    )
    return parser.parse_args()


def _to_fp16_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        converted[key] = tensor.half() if tensor.is_floating_point() else tensor
    return converted


def main() -> None:
    args = parse_args()
    input_path = resolve_cli_path(args.input)
    output_path = resolve_cli_path(args.output)

    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing a 'state_dict' entry.")

    exported = dict(checkpoint)
    exported["state_dict"] = _to_fp16_state_dict(checkpoint["state_dict"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)

    source_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Exported fp16 checkpoint to {output_path} | "
        f"source={source_size:.2f} MiB | output={output_size:.2f} MiB"
    )


if __name__ == "__main__":
    main()
