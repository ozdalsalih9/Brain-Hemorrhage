from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedKFold, train_test_split


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _numeric_sort_key(value: str) -> tuple[int, Any]:
    value = str(value).strip()
    return (0, int(value)) if value.isdigit() else (1, value.lower())


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip() for column in df.columns]
    return df


def _find_image_root(dataset_root: Path) -> Path | None:
    best_dir = None
    best_count = 0
    for directory in [dataset_root] + [path for path in dataset_root.rglob("*") if path.is_dir()]:
        count = sum(1 for child in directory.iterdir() if is_image_file(child))
        if count > best_count:
            best_dir = directory
            best_count = count
    return best_dir


def _build_image_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in sorted(image_root.rglob("*")):
        if not is_image_file(image_path):
            continue
        index[image_path.stem] = image_path
        index[image_path.name] = image_path
    return index


def _label_name_mapping(label_column: str, raw_values: list[str]) -> dict[str, str]:
    unique_values = sorted({str(value).strip() for value in raw_values}, key=_numeric_sort_key)
    if set(unique_values) <= {"0", "1"}:
        if label_column.lower() == "hemorrhage":
            return {"0": "no_hemorrhage", "1": "hemorrhage"}
        return {value: f"class_{value}" for value in unique_values}
    return {value: value.strip().replace(" ", "_").lower() for value in unique_values}


def _infer_label_column(df: pd.DataFrame) -> str:
    preferred = ("label", "class", "target", "diagnosis", "hemorrhage")
    lower_map = {column.lower(): column for column in df.columns}
    for column in preferred:
        if column in lower_map:
            return lower_map[column]

    excluded = {"id", "image", "image_id", "path", "image_path", "filepath", "filename", "file"}
    candidates = [column for column in df.columns if column.lower() not in excluded]
    if len(candidates) != 1:
        raise ValueError(
            "Label column could not be inferred automatically. "
            f"Columns found: {', '.join(df.columns)}"
        )
    return candidates[0]


def _infer_path_column(df: pd.DataFrame) -> str | None:
    for column in ("image_path", "path", "filepath", "filename", "file", "image"):
        if column in df.columns:
            return column
    return None


def _verify_image(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return None
    except (OSError, UnidentifiedImageError) as exc:
        return str(exc)


def _analyze_csv_dataset(dataset_root: Path, labels_csv: Path) -> dict[str, Any]:
    df = _sanitize_columns(pd.read_csv(labels_csv))
    label_column = _infer_label_column(df)
    path_column = _infer_path_column(df)
    image_root = _find_image_root(dataset_root)
    if image_root is None:
        raise FileNotFoundError(f"No image files were found under {dataset_root}")

    image_index = _build_image_index(image_root)
    label_map = _label_name_mapping(label_column, df[label_column].astype(str).tolist())

    samples: list[dict[str, Any]] = []
    missing_files: list[str] = []
    broken_files: list[dict[str, str]] = []

    for row in df.to_dict(orient="records"):
        raw_label = str(row[label_column]).strip()
        label_name = label_map[raw_label]

        image_path: Path | None = None
        if path_column:
            candidate = dataset_root / str(row[path_column])
            if candidate.exists():
                image_path = candidate
            else:
                candidate = image_root / str(row[path_column])
                if candidate.exists():
                    image_path = candidate
        elif "id" in row:
            raw_id = str(row["id"]).strip()
            candidate_keys = [raw_id]
            if raw_id.isdigit():
                candidate_keys.extend([str(int(raw_id)), f"{int(raw_id):03d}", f"{int(raw_id):04d}"])
            for key in candidate_keys:
                if key in image_index:
                    image_path = image_index[key]
                    break

        if image_path is None:
            missing_files.append(str(row.get(path_column or "id", "unknown")))
            continue

        image_error = _verify_image(image_path)
        if image_error is not None:
            broken_files.append({"path": str(image_path), "error": image_error})
            continue

        samples.append(
            {
                "image_path": str(image_path),
                "label_raw": raw_label,
                "label_name": label_name,
            }
        )

    class_names = [label_map[value] for value in sorted(label_map.keys(), key=_numeric_sort_key)]
    class_counts = Counter(sample["label_name"] for sample in samples)
    return {
        "layout": "csv",
        "dataset_root": str(dataset_root.resolve()),
        "labels_csv": str(labels_csv.resolve()),
        "image_root": str(image_root.resolve()),
        "label_column": label_column,
        "total_records": int(len(df)),
        "total_images": int(len(samples)),
        "class_names": class_names,
        "class_counts": dict(class_counts),
        "missing_files": missing_files,
        "broken_files": broken_files,
        "samples": samples,
    }


def _analyze_folder_dataset(dataset_root: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    broken_files: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()

    class_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name.lower() not in SPLIT_NAMES],
        key=lambda path: path.name.lower(),
    )
    if not class_dirs:
        raise FileNotFoundError(
            "No folder-based classes found. Expected class folders or a labels.csv file."
        )

    for class_dir in class_dirs:
        class_name = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if not is_image_file(image_path):
                continue
            image_error = _verify_image(image_path)
            if image_error is not None:
                broken_files.append({"path": str(image_path), "error": image_error})
                continue
            samples.append(
                {
                    "image_path": str(image_path),
                    "label_raw": class_name,
                    "label_name": class_name,
                }
            )
            class_counts[class_name] += 1

    return {
        "layout": "folder",
        "dataset_root": str(dataset_root.resolve()),
        "labels_csv": None,
        "image_root": str(dataset_root.resolve()),
        "label_column": "folder_name",
        "total_records": int(len(samples)),
        "total_images": int(len(samples)),
        "class_names": [path.name for path in class_dirs],
        "class_counts": dict(class_counts),
        "missing_files": [],
        "broken_files": broken_files,
        "samples": samples,
    }


def _analyze_pre_split_dataset(dataset_root: Path) -> dict[str, Any]:
    split_aliases = {"validation": "val"}
    class_counts: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    broken_files: list[dict[str, str]] = []
    class_names: set[str] = set()

    for split_dir in sorted([path for path in dataset_root.iterdir() if path.is_dir()]):
        split_name = split_aliases.get(split_dir.name.lower(), split_dir.name.lower())
        if split_name not in SPLIT_NAMES:
            continue
        for class_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
            class_name = class_dir.name
            class_names.add(class_name)
            for image_path in sorted(class_dir.rglob("*")):
                if not is_image_file(image_path):
                    continue
                image_error = _verify_image(image_path)
                if image_error is not None:
                    broken_files.append({"path": str(image_path), "error": image_error})
                    continue
                sample = {
                    "image_path": str(image_path),
                    "label_raw": class_name,
                    "label_name": class_name,
                }
                samples_by_split[split_name].append(sample)
                split_counts[split_name][class_name] += 1
                class_counts[class_name] += 1

    total_images = sum(len(samples) for samples in samples_by_split.values())
    if total_images == 0:
        raise FileNotFoundError("Pre-split dataset structure detected, but no image files were found.")

    return {
        "layout": "folder_split",
        "dataset_root": str(dataset_root.resolve()),
        "labels_csv": None,
        "image_root": str(dataset_root.resolve()),
        "label_column": "folder_name",
        "total_records": int(total_images),
        "total_images": int(total_images),
        "class_names": sorted(class_names),
        "class_counts": dict(class_counts),
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "missing_files": [],
        "broken_files": broken_files,
        "samples": [],
        "samples_by_split": dict(samples_by_split),
    }


def analyze_dataset(dataset_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    split_dirs = {path.name.lower() for path in dataset_root.iterdir() if path.is_dir()}
    if {"train", "test"} <= split_dirs and ("val" in split_dirs or "validation" in split_dirs):
        return _analyze_pre_split_dataset(dataset_root)

    labels_csv = dataset_root / "labels.csv"
    if labels_csv.exists():
        return _analyze_csv_dataset(dataset_root, labels_csv)

    return _analyze_folder_dataset(dataset_root)


def _make_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_analysis_reports(analysis: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = ensure_dir(Path(output_dir))
    analysis_json_path = output_dir / "dataset_analysis.json"
    analysis_txt_path = output_dir / "dataset_analysis.txt"

    json_ready = {
        key: value
        for key, value in analysis.items()
        if key not in {"samples", "samples_by_split"}
    }
    with analysis_json_path.open("w", encoding="utf-8") as file:
        json.dump(json_ready, file, indent=2, ensure_ascii=False)

    lines = [
        f"Layout: {analysis['layout']}",
        f"Dataset root: {analysis['dataset_root']}",
        f"Total images: {analysis['total_images']}",
        "Class counts:",
    ]
    for class_name, count in analysis["class_counts"].items():
        lines.append(f"  - {class_name}: {count}")

    if "split_counts" in analysis:
        lines.append("Existing split counts:")
        for split_name, counts in analysis["split_counts"].items():
            summary = ", ".join(f"{class_name}={count}" for class_name, count in counts.items())
            lines.append(f"  - {split_name}: {summary}")

    lines.append(f"Missing files: {len(analysis['missing_files'])}")
    lines.append(f"Broken files: {len(analysis['broken_files'])}")
    if analysis["missing_files"]:
        lines.append("Missing file examples:")
        lines.extend(f"  - {item}" for item in analysis["missing_files"][:10])
    if analysis["broken_files"]:
        lines.append("Broken file examples:")
        lines.extend(f"  - {item['path']} :: {item['error']}" for item in analysis["broken_files"][:10])

    analysis_txt_path.write_text("\n".join(lines), encoding="utf-8")
    return analysis_json_path, analysis_txt_path


def _write_manifest(
    samples: list[dict[str, Any]],
    dataset_root: Path,
    class_to_idx: dict[str, int],
    manifest_path: Path,
) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for sample in samples:
        label_name = sample["label_name"]
        rows.append(
            {
                "image_path": _make_relative(Path(sample["image_path"]), dataset_root),
                "label_name": label_name,
                "label_raw": sample["label_raw"],
                "label_idx": class_to_idx[label_name],
            }
        )
        counts[label_name] += 1

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return dict(counts)


def _save_split_metadata(
    split_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    split_counts: dict[str, dict[str, int]],
    analysis: dict[str, Any],
) -> Path:
    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "class_names": class_names,
        "class_to_idx": {class_name: index for index, class_name in enumerate(class_names)},
        "split_counts": split_counts,
        "analysis_layout": analysis["layout"],
    }
    metadata_path = split_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return metadata_path


def create_stratified_split_manifests(
    analysis: dict[str, Any],
    split_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    split_dir = ensure_dir(Path(split_dir))
    dataset_root = Path(analysis["dataset_root"]).resolve()

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    class_names = analysis["class_names"]
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    split_counts: dict[str, dict[str, int]] = {}

    if analysis["layout"] == "folder_split":
        for split_name in SPLIT_NAMES:
            samples = analysis["samples_by_split"].get(split_name, [])
            split_counts[split_name] = _write_manifest(
                samples=samples,
                dataset_root=dataset_root,
                class_to_idx=class_to_idx,
                manifest_path=split_dir / f"{split_name}.csv",
            )
        metadata_path = _save_split_metadata(split_dir, dataset_root, class_names, split_counts, analysis)
        return {"split_dir": str(split_dir), "metadata_path": str(metadata_path), "split_counts": split_counts}

    samples_df = pd.DataFrame(analysis["samples"])
    train_df, temp_df = train_test_split(
        samples_df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=samples_df["label_name"],
    )

    temp_ratio = val_ratio + test_ratio
    val_share = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=temp_df["label_name"],
    )

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, frame in split_frames.items():
        split_counts[split_name] = _write_manifest(
            samples=frame.to_dict(orient="records"),
            dataset_root=dataset_root,
            class_to_idx=class_to_idx,
            manifest_path=split_dir / f"{split_name}.csv",
        )

    metadata_path = _save_split_metadata(split_dir, dataset_root, class_names, split_counts, analysis)
    return {"split_dir": str(split_dir), "metadata_path": str(metadata_path), "split_counts": split_counts}


def create_stratified_kfold_manifests(
    analysis: dict[str, Any],
    output_dir: str | Path,
    folds: int = 5,
    val_ratio_within_train: float = 0.125,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")

    output_dir = ensure_dir(Path(output_dir))
    dataset_root = Path(analysis["dataset_root"]).resolve()
    class_names = analysis["class_names"]
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}

    if analysis["layout"] == "folder_split":
        samples: list[dict[str, Any]] = []
        for split_name in SPLIT_NAMES:
            samples.extend(analysis.get("samples_by_split", {}).get(split_name, []))
    else:
        samples = list(analysis["samples"])

    samples_df = pd.DataFrame(samples)
    if samples_df.empty:
        raise ValueError("No samples were available to build k-fold manifests.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, Any]] = []
    labels = samples_df["label_name"]

    for fold_index, (train_val_indices, test_indices) in enumerate(splitter.split(samples_df, labels), start=1):
        fold_dir = ensure_dir(output_dir / f"fold_{fold_index:02d}")
        train_val_df = samples_df.iloc[train_val_indices].reset_index(drop=True)
        test_df = samples_df.iloc[test_indices].reset_index(drop=True)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_within_train,
            random_state=seed + fold_index,
            stratify=train_val_df["label_name"],
        )

        split_frames = {"train": train_df, "val": val_df, "test": test_df}
        split_counts: dict[str, dict[str, int]] = {}
        for split_name, frame in split_frames.items():
            split_counts[split_name] = _write_manifest(
                samples=frame.to_dict(orient="records"),
                dataset_root=dataset_root,
                class_to_idx=class_to_idx,
                manifest_path=fold_dir / f"{split_name}.csv",
            )

        metadata_path = _save_split_metadata(
            split_dir=fold_dir,
            dataset_root=dataset_root,
            class_names=class_names,
            split_counts=split_counts,
            analysis=analysis,
        )
        fold_results.append(
            {
                "fold_index": fold_index,
                "split_dir": str(fold_dir.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "split_counts": split_counts,
            }
        )

    return fold_results


def load_split_metadata(split_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(split_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_dataset(
    dataset_root: str | Path,
    split_dir: str | Path,
    results_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    split_dir = Path(split_dir).resolve()
    results_dir = Path(results_dir).resolve()

    analysis = analyze_dataset(dataset_root)
    analysis_json, analysis_txt = write_analysis_reports(analysis, results_dir)

    split_paths = [split_dir / f"{split_name}.csv" for split_name in SPLIT_NAMES]
    metadata_path = split_dir / "metadata.json"
    manifests_exist = metadata_path.exists() and all(path.exists() for path in split_paths)

    if force or not manifests_exist:
        split_result = create_stratified_split_manifests(
            analysis=analysis,
            split_dir=split_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        split_result = {"split_dir": str(split_dir), "metadata_path": str(metadata_path)}

    metadata = load_split_metadata(split_dir)
    return {
        "analysis": analysis,
        "analysis_json": str(analysis_json),
        "analysis_txt": str(analysis_txt),
        "split_dir": split_result["split_dir"],
        "metadata_path": split_result["metadata_path"],
        "metadata": metadata,
    }
