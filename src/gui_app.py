from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

try:
    from model_utils import (
        analyze_scan,
        collect_prediction_records,
        create_manifest_loader,
        estimate_prediction_accuracy,
        explain_prediction,
        get_device,
        load_checkpoint,
        resolve_image_size,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_utils import (
        analyze_scan,
        collect_prediction_records,
        create_manifest_loader,
        estimate_prediction_accuracy,
        explain_prediction,
        get_device,
        load_checkpoint,
        resolve_image_size,
    )


WINDOW_TITLE = "Brain CT Hemorrhage Detection"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_SIZE = (380, 380)
MODEL_PREVIEW_SIZE = (210, 190)

BG = "#f3efe7"
PANEL = "#ffffff"
PANEL_ALT = "#f7f9fc"
TILE = "#eef3f8"
EDGE = "#d8e0ea"
TEXT = "#182433"
MUTED = "#66768a"
ACCENT = "#0f7c82"
SUCCESS = "#1f8a5b"
WARNING = "#cf6433"
SUBTLE = "#708196"
CUSTOM = "#bd8a17"


def _make_preview(image: Image.Image, size: tuple[int, int] = PREVIEW_SIZE) -> ImageTk.PhotoImage:
    preview = image.copy()
    preview.thumbnail(size)
    canvas = Image.new("RGB", size, color=(238, 243, 248))
    paste_x = (size[0] - preview.width) // 2
    paste_y = (size[1] - preview.height) // 2
    canvas.paste(preview, (paste_x, paste_y))
    return ImageTk.PhotoImage(canvas)


def _risk_label(predicted_class: str, confidence: float) -> tuple[str, str]:
    if predicted_class != "hemorrhage":
        return ("Stable", SUCCESS) if confidence >= 0.85 else ("Low Risk", ACCENT)
    if confidence >= 0.90:
        return "Critical", WARNING
    if confidence >= 0.75:
        return "Elevated", WARNING
    return "Needs Review", ACCENT


def _format_probability_text(probabilities: dict[str, float]) -> str:
    hemorrhage = float(probabilities.get("hemorrhage", 0.0))
    no_hemorrhage = float(probabilities.get("no_hemorrhage", 0.0))
    return f"Hemorrhage: {hemorrhage * 100:.2f}%   |   No hemorrhage: {no_hemorrhage * 100:.2f}%"


def _read_checkpoint_model_name(checkpoint_path: Path) -> str | None:
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(checkpoint, dict):
        return None
    model_name = checkpoint.get("model_name")
    return str(model_name).lower() if model_name is not None else None


class PredictionApp:
    def __init__(self, root: tk.Tk, checkpoint_paths: dict[str, Path]) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1500x900")
        self.root.minsize(1260, 760)
        self.root.configure(bg=BG)

        self.device = get_device()
        self.models = self._load_models(checkpoint_paths)
        if "pretrained" not in self.models:
            raise FileNotFoundError("Pretrained checkpoint could not be loaded.")

        self.selected_path = tk.StringVar()
        loaded_names = " | ".join(
            f"{info['title']}: {info['checkpoint_path'].name}" for info in self.models.values()
        )
        self.status_text = tk.StringVar(value=f"{loaded_names}   |   Device: {self.device.type.upper()}")
        self.analysis_text = tk.StringVar(
            value="Upload a CT image to compare the pretrained model and the custom CNN on the same scan."
        )
        self.summary_text = tk.StringVar(value="Awaiting scan")
        self.current_scan_info: dict | None = None
        self.original_photo: ImageTk.PhotoImage | None = None
        self.model_overlay_photos: dict[str, ImageTk.PhotoImage | None] = {}
        self.is_animating = False
        self.metric_labels: dict[str, tk.Label] = {}
        self.model_result_labels: dict[str, dict[str, tk.Label]] = {}
        self.model_text_vars: dict[str, dict[str, tk.StringVar]] = {}

        self._configure_style()
        self._build_ui()
        self._reset_model_cards()

    def _load_models(self, checkpoint_paths: dict[str, Path]) -> dict[str, dict]:
        models: dict[str, dict] = {}
        model_specs = {
            "pretrained": {"title": "Pretrained CNN", "accent": ACCENT},
            "custom": {"title": "Custom CNN", "accent": CUSTOM},
        }
        for key, path in checkpoint_paths.items():
            if not path.exists():
                continue
            model, checkpoint = load_checkpoint(path, device=self.device)
            models[key] = {
                "model": model,
                "checkpoint": checkpoint,
                "checkpoint_path": path,
                "title": model_specs.get(key, {}).get("title", key.title()),
                "accent": model_specs.get(key, {}).get("accent", ACCENT),
                "analysis_reference": self._build_analysis_reference(model, checkpoint),
                "evaluation_summary": self._load_evaluation_summary(path),
            }
        return models

    def _load_evaluation_summary(self, checkpoint_path: Path) -> dict | None:
        results_root = PROJECT_ROOT / "results"
        if not results_root.exists():
            return None

        checkpoint_name = checkpoint_path.name.lower()
        for summary_path in results_root.glob("*/training_summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            recorded_checkpoint = summary.get("checkpoint")
            if not recorded_checkpoint:
                continue

            recorded_name = Path(str(recorded_checkpoint)).name.lower()
            if recorded_name == checkpoint_name:
                return summary

        return None

    def _build_analysis_reference(self, model, checkpoint: dict) -> list[dict]:
        dataset_root = PROJECT_ROOT / "Dataset"
        split_dir = dataset_root / "splits"
        manifests = [split_dir / "val.csv", split_dir / "test.csv"]
        records: list[dict] = []

        for manifest_path in manifests:
            if not manifest_path.exists():
                continue
            loader = create_manifest_loader(
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                batch_size=8,
                image_size=resolve_image_size(checkpoint.get("image_size"), checkpoint.get("model_name")),
                model_name=checkpoint.get("model_name"),
                mean=checkpoint.get("normalization_mean"),
                std=checkpoint.get("normalization_std"),
                shuffle=False,
            )
            records.extend(
                collect_prediction_records(
                    model=model,
                    data_loader=loader,
                    device=self.device,
                    class_names=checkpoint["class_names"],
                )
            )
        return records

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor="#dce6ee",
            background=ACCENT,
            bordercolor="#dce6ee",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        style.configure("App.TNotebook", background=PANEL, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure(
            "App.TNotebook.Tab",
            font=("Segoe UI", 9, "bold"),
            padding=(12, 6),
            background=TILE,
            foreground=TEXT,
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", PANEL_ALT)],
            foreground=[("selected", ACCENT)],
        )

    def _card(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE, highlightthickness=1, bd=0)
        tk.Label(
            frame,
            text=title,
            bg=PANEL,
            fg=ACCENT,
            font=("Bahnschrift SemiBold", 12),
            anchor="w",
        ).pack(fill="x", padx=22, pady=(18, 12))
        return frame

    def _metric_tile(
        self,
        parent: tk.Widget,
        title: str,
        variable: tk.StringVar,
        value_color: str = TEXT,
    ) -> tk.Frame:
        tile = tk.Frame(parent, bg=TILE, highlightbackground=EDGE, highlightthickness=1)
        tk.Label(
            tile,
            text=title,
            bg=TILE,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 3))
        tk.Label(
            tile,
            textvariable=variable,
            bg=TILE,
            fg=value_color,
            font=("Bahnschrift SemiBold", 12),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 9))
        return tile

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 18))
        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Brain CT Hemorrhage Detection",
            bg=BG,
            fg=TEXT,
            font=("Bahnschrift SemiBold", 30),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Upload a scan, compare both models, and inspect the suspicious region with a larger visual focus box.",
            bg=BG,
            fg=SUBTLE,
            font=("Segoe UI", 11),
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        content = tk.Frame(self.root, bg=BG)
        content.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 16))
        content.grid_columnconfigure(0, weight=4, minsize=460)
        content.grid_columnconfigure(1, weight=7, minsize=760)
        content.grid_rowconfigure(0, weight=1)

        self._build_left_panel(content)
        self._build_right_panel(content)

        status = tk.Label(
            self.root,
            textvariable=self.status_text,
            bg=BG,
            fg=MUTED,
            font=("Consolas", 10),
            anchor="w",
        )
        status.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 18))

    def _build_left_panel(self, parent: tk.Widget) -> None:
        panel = self._card(parent, "Scan Workspace")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=22, pady=(0, 22))

        path_row = tk.Frame(inner, bg=PANEL)
        path_row.pack(fill="x")

        self.path_entry = tk.Entry(
            path_row,
            textvariable=self.selected_path,
            relief="flat",
            bd=0,
            readonlybackground=TILE,
            fg=TEXT,
            font=("Segoe UI", 11),
        )
        self.path_entry.pack(side="left", fill="x", expand=True, ipady=11, padx=(0, 12))
        self.path_entry.configure(state="readonly")

        self.browse_button = tk.Button(
            path_row,
            text="Browse",
            command=self.select_image,
            bg=TEXT,
            fg="#ffffff",
            activebackground="#243244",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=24,
            pady=11,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.browse_button.pack(side="right")

        actions_row = tk.Frame(inner, bg=PANEL)
        actions_row.pack(fill="x", pady=(12, 0))

        self.compare_button = tk.Button(
            actions_row,
            text="Compare",
            command=self.run_prediction,
            bg=TEXT,
            fg="#ffffff",
            activebackground="#243244",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=18,
            pady=11,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.compare_button.pack(side="left")

        tk.Label(
            inner,
            text="Accepted formats: PNG, JPG, JPEG, BMP, TIFF",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(12, 18))

        summary_card = tk.Frame(inner, bg=PANEL_ALT, highlightbackground=EDGE, highlightthickness=1)
        summary_card.pack(fill="x", pady=(0, 18))
        tk.Label(
            summary_card,
            text="SESSION STATUS",
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(12, 2))
        tk.Label(
            summary_card,
            textvariable=self.summary_text,
            bg=PANEL_ALT,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
            justify="left",
            wraplength=430,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 12))

        tk.Label(
            inner,
            text="Original Scan Preview",
            bg=PANEL,
            fg=TEXT,
            font=("Bahnschrift SemiBold", 13),
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        preview_frame = tk.Frame(inner, bg=TILE, highlightbackground=EDGE, highlightthickness=1, height=PREVIEW_SIZE[1] + 24)
        preview_frame.pack(fill="both", expand=True)
        preview_frame.pack_propagate(False)

        self.preview_label = tk.Label(
            preview_frame,
            text="No image selected",
            bg=TILE,
            fg=MUTED,
            font=("Segoe UI", 12, "bold"),
        )
        self.preview_label.pack(fill="both", expand=True, padx=16, pady=16)

        controls = tk.Frame(inner, bg=PANEL)
        controls.pack(fill="x", pady=(18, 0))

        self.clear_button = tk.Button(
            controls,
            text="Clear",
            command=self.clear_selection,
            bg=TILE,
            fg=TEXT,
            activebackground="#dfe7ef",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=28,
            pady=12,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.clear_button.pack(side="left")

    def _build_model_card(self, parent: tk.Widget, model_key: str, title: str, accent: str) -> tk.Frame:
        card = tk.Frame(parent, bg=PANEL_ALT, highlightbackground=EDGE, highlightthickness=1)
        card.grid_columnconfigure(0, weight=1)

        vars_for_model = {
            "result": tk.StringVar(value="Awaiting scan"),
            "hemorrhage_prob": tk.StringVar(value="--"),
            "no_hemorrhage_prob": tk.StringVar(value="--"),
            "confidence": tk.StringVar(value="--"),
            "analysis_accuracy": tk.StringVar(value="--"),
            "region": tk.StringVar(value="Hemorrhage location: --"),
            "checkpoint": tk.StringVar(value="Checkpoint: --"),
        }
        self.model_text_vars[model_key] = vars_for_model

        header = tk.Frame(card, bg=PANEL_ALT)
        header.pack(fill="x", padx=16, pady=(14, 0))

        tk.Label(
            header,
            text=title,
            bg=PANEL_ALT,
            fg=accent,
            font=("Bahnschrift SemiBold", 14),
            anchor="w",
        ).pack(anchor="w")
        checkpoint_label = tk.Label(
            card,
            textvariable=vars_for_model["checkpoint"],
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Consolas", 8),
            anchor="w",
            justify="left",
            wraplength=330,
        )
        checkpoint_label.pack(fill="x", padx=16, pady=(4, 0))
        result_label = tk.Label(
            card,
            textvariable=vars_for_model["result"],
            bg=PANEL_ALT,
            fg=TEXT,
            font=("Bahnschrift SemiBold", 18),
            anchor="w",
            justify="left",
            wraplength=330,
        )
        result_label.pack(fill="x", padx=16, pady=(10, 6))

        metrics = tk.Frame(card, bg=PANEL_ALT)
        metrics.pack(fill="x", padx=16)
        metrics.grid_columnconfigure(0, weight=1, uniform="metric")
        metrics.grid_columnconfigure(1, weight=1, uniform="metric")

        hemorrhage_tile = self._metric_tile(metrics, "Hemorrhage", vars_for_model["hemorrhage_prob"], WARNING)
        hemorrhage_tile.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        no_hemorrhage_tile = self._metric_tile(
            metrics, "No Hemorrhage", vars_for_model["no_hemorrhage_prob"], SUCCESS
        )
        no_hemorrhage_tile.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 8))

        confidence_tile = self._metric_tile(metrics, "Confidence", vars_for_model["confidence"], TEXT)
        confidence_tile.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        analysis_accuracy_tile = self._metric_tile(
            metrics, "Est. Reliability", vars_for_model["analysis_accuracy"], ACCENT
        )
        analysis_accuracy_tile.grid(row=1, column=1, sticky="ew", padx=(6, 0))

        region_label = tk.Label(
            card,
            textvariable=vars_for_model["region"],
            bg=PANEL_ALT,
            fg=SUCCESS,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify="left",
            wraplength=330,
        )
        if model_key == "pretrained":
            region_label.pack(fill="x", padx=16, pady=(10, 10))

        self.model_result_labels[model_key] = {
            "result": result_label,
            "checkpoint": checkpoint_label,
        }
        if model_key == "pretrained":
            self.model_result_labels[model_key]["region"] = region_label
        return card

    def _build_right_panel(self, parent: tk.Widget) -> None:
        panel = self._card(parent, "Live Comparison")
        panel.grid(row=0, column=1, sticky="nsew", padx=(14, 0))

        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=22, pady=(0, 22))

        models_frame = tk.Frame(inner, bg=PANEL)
        models_frame.pack(fill="both", expand=True)
        models_frame.grid_columnconfigure(0, weight=1)
        models_frame.grid_columnconfigure(1, weight=1)
        models_frame.grid_rowconfigure(0, weight=1)

        pretrained_card = self._build_model_card(models_frame, "pretrained", "Pretrained CNN", ACCENT)
        pretrained_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        custom_card = self._build_model_card(models_frame, "custom", "Custom CNN", CUSTOM)
        custom_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        notebook = ttk.Notebook(inner, style="App.TNotebook")
        notebook.pack(fill="both", expand=True, pady=(12, 0))

        localization_tab = tk.Frame(notebook, bg=PANEL)
        diagnostics_tab = tk.Frame(notebook, bg=PANEL)
        notebook.add(localization_tab, text="Localization")
        notebook.add(diagnostics_tab, text="Diagnostics")

        overlay_card = tk.Frame(localization_tab, bg=PANEL_ALT, highlightbackground=EDGE, highlightthickness=1)
        overlay_card.pack(fill="both", expand=True)
        overlay_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            overlay_card,
            text="PRETRAINED LOCALIZATION",
            bg=PANEL_ALT,
            fg=ACCENT,
            font=("Bahnschrift SemiBold", 12),
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))

        overlay_frame = tk.Frame(
            overlay_card,
            bg=TILE,
            highlightbackground=EDGE,
            highlightthickness=1,
            width=MODEL_PREVIEW_SIZE[0] + 18,
            height=MODEL_PREVIEW_SIZE[1] + 18,
        )
        overlay_frame.grid(row=1, column=0, padx=14, pady=(0, 12))
        overlay_frame.grid_propagate(False)

        self.pretrained_overlay_label = tk.Label(
            overlay_frame,
            text="Localization preview unavailable",
            bg=TILE,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
            anchor="center",
            justify="center",
        )
        self.pretrained_overlay_label.pack(expand=True, padx=6, pady=6)

        lower_frame = tk.Frame(diagnostics_tab, bg=PANEL)
        lower_frame.pack(fill="both", expand=True)
        lower_frame.grid_columnconfigure(0, weight=4, uniform="bottom")
        lower_frame.grid_columnconfigure(1, weight=6, uniform="bottom")

        diagnostics = tk.Frame(lower_frame, bg=PANEL_ALT, highlightbackground=EDGE, highlightthickness=1)
        diagnostics.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        diagnostics.grid_columnconfigure(1, weight=1)

        tk.Label(
            diagnostics,
            text="SCAN DIAGNOSTICS",
            bg=PANEL_ALT,
            fg=ACCENT,
            font=("Bahnschrift SemiBold", 12),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 8))

        metric_names = ["Agreement", "Risk Band", "Image Size", "Contrast", "Density Pattern", "Framing"]
        for row, name in enumerate(metric_names, start=1):
            tk.Label(
                diagnostics,
                text=name,
                bg=PANEL_ALT,
                fg=SUBTLE,
                font=("Segoe UI", 10),
                anchor="w",
            ).grid(row=row, column=0, sticky="nw", padx=14, pady=5)

            value = tk.Label(
                diagnostics,
                text="--",
                bg=PANEL_ALT,
                fg=TEXT,
                font=("Segoe UI", 10, "bold"),
                anchor="w",
                justify="left",
                wraplength=220,
            )
            value.grid(row=row, column=1, sticky="nw", pady=5, padx=(8, 14))
            self.metric_labels[name] = value

        analysis_card = tk.Frame(lower_frame, bg=PANEL_ALT, highlightbackground=EDGE, highlightthickness=1)
        analysis_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        analysis_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            analysis_card,
            text="ANALYSIS BRIEF",
            bg=PANEL_ALT,
            fg=ACCENT,
            font=("Bahnschrift SemiBold", 12),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        self.analysis_message = tk.Message(
            analysis_card,
            textvariable=self.analysis_text,
            bg=PANEL_ALT,
            fg=TEXT,
            font=("Segoe UI", 10),
            width=420,
            justify="left",
            padx=14,
            pady=6,
        )
        self.analysis_message.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.progress = ttk.Progressbar(
            analysis_card,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            style="Custom.Horizontal.TProgressbar",
        )
        self.progress.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.progress.grid_remove()

    def _reset_model_cards(self) -> None:
        for model_key, model_info in self.models.items():
            vars_for_model = self.model_text_vars[model_key]
            vars_for_model["result"].set("Awaiting scan")
            vars_for_model["hemorrhage_prob"].set("--")
            vars_for_model["no_hemorrhage_prob"].set("--")
            vars_for_model["confidence"].set("--")
            vars_for_model["analysis_accuracy"].set("--")
            vars_for_model["region"].set("Hemorrhage location: --")
            vars_for_model["checkpoint"].set(f"Checkpoint: {model_info['checkpoint_path'].name}")
            self.model_overlay_photos[model_key] = None
            self.model_result_labels[model_key]["result"].configure(fg=TEXT)
        self.pretrained_overlay_label.configure(image="", text="Localization preview unavailable")

        if "custom" not in self.models:
            self.model_text_vars["custom"]["result"].set("Checkpoint unavailable")
            self.model_text_vars["custom"]["hemorrhage_prob"].set("--")
            self.model_text_vars["custom"]["no_hemorrhage_prob"].set("--")
            self.model_text_vars["custom"]["analysis_accuracy"].set("--")
            self.model_text_vars["custom"]["checkpoint"].set("Checkpoint: not found in models folder")

    def _update_scan_metrics(self, scan_info: dict, agreement_text: str = "--", risk_text: str = "--") -> None:
        self.metric_labels["Agreement"].configure(text=agreement_text, fg=TEXT)
        risk_color = WARNING if "positive" in risk_text.lower() else ACCENT if "mixed" in risk_text.lower() else SUCCESS
        if risk_text in {"--", "Not a brain CT image"}:
            risk_color = WARNING if risk_text != "--" else TEXT
        if "atypical" in risk_text.lower():
            risk_color = WARNING
        self.metric_labels["Risk Band"].configure(text=risk_text, fg=risk_color)
        self.metric_labels["Image Size"].configure(text=f"{scan_info['width']} x {scan_info['height']} px", fg=TEXT)
        self.metric_labels["Contrast"].configure(text=f"{scan_info['contrast']:.3f}", fg=TEXT)
        self.metric_labels["Density Pattern"].configure(text=scan_info["density_pattern"], fg=TEXT)
        self.metric_labels["Framing"].configure(text=scan_info["framing"], fg=TEXT)

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select CT Image",
            initialdir=str(PROJECT_ROOT / "Dataset"),
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
        )
        if not file_path:
            return

        selected = Path(file_path)
        self.selected_path.set(str(selected))

        original = Image.open(selected).convert("RGB")
        original = ImageOps.autocontrast(original.convert("L")).convert("RGB")
        self.original_photo = _make_preview(original)
        self.preview_label.configure(image=self.original_photo, text="")
        self.current_scan_info = analyze_scan(selected)

        self._reset_model_cards()
        self._update_scan_metrics(self.current_scan_info)

        if self.current_scan_info["brain_ct_should_block"]:
            self.compare_button.configure(state="disabled")
            self.summary_text.set("Input rejected. Uploaded image does not appear to be a brain CT slice.")
            self.analysis_text.set(self.current_scan_info["brain_ct_message"])
            self._update_scan_metrics(
                self.current_scan_info,
                agreement_text="Input rejected",
                risk_text="Not a brain CT image",
            )
            self.status_text.set(f"Rejected input: {selected.name}")
            messagebox.showwarning("Invalid image", self.current_scan_info["brain_ct_message"])
        elif self.current_scan_info["brain_ct_status"] == "uncertain":
            self.compare_button.configure(state="normal")
            self.summary_text.set("Scan loaded with caution. The image may be a brain CT, but framing or intensity is atypical.")
            self.analysis_text.set(self.current_scan_info["brain_ct_message"])
            self._update_scan_metrics(
                self.current_scan_info,
                agreement_text="Input warning",
                risk_text="Atypical but allowed",
            )
            self.status_text.set(f"Caution: {selected.name}")
        else:
            self.compare_button.configure(state="normal")
            self.summary_text.set("Scan ready. Run both models to compare probabilities, estimated reliability, and localized focus.")
            self.analysis_text.set("The selected scan is ready. Both models will analyze the same image and render their own focus map.")
            self.status_text.set(f"Ready: {selected.name}")

    def clear_selection(self) -> None:
        if self.is_animating:
            return

        self.selected_path.set("")
        self.current_scan_info = None
        self.preview_label.configure(image="", text="No image selected")
        self.original_photo = None
        self.summary_text.set("Awaiting scan")
        self.analysis_text.set("Upload a CT image to compare the pretrained model and the custom CNN on the same scan.")
        self.progress.grid_remove()
        self.progress["value"] = 0
        self._reset_model_cards()
        self.compare_button.configure(state="normal", text="Compare")
        for label in self.metric_labels.values():
            label.configure(text="--", fg=TEXT)
        loaded_names = " | ".join(
            f"{info['title']}: {info['checkpoint_path'].name}" for info in self.models.values()
        )
        self.status_text.set(f"{loaded_names}   |   Device: {self.device.type.upper()}")

    def run_prediction(self) -> None:
        if not self.selected_path.get():
            messagebox.showwarning("Missing file", "Please select an image first.")
            return
        if self.current_scan_info is not None and self.current_scan_info["brain_ct_should_block"]:
            messagebox.showwarning("Invalid image", "Uploaded image does not appear to be a brain CT image.")
            return
        if self.is_animating:
            return

        self.is_animating = True
        self.browse_button.configure(state="disabled")
        self.compare_button.configure(state="disabled")
        self.clear_button.configure(state="disabled")
        self.progress["value"] = 0
        self.progress.grid()
        self._animate_progress(0)

    def _animate_progress(self, step: int) -> None:
        phases = ["Analyzing.", "Analyzing..", "Analyzing..."]
        total_steps = 10
        self.summary_text.set(
            f"{phases[step % len(phases)]} Running pretrained and custom CNN predictions on the uploaded scan."
        )
        self.analysis_text.set(
            "Both models are being executed, focal text estimates are prepared, and shared scan diagnostics are being generated."
        )
        self.progress["value"] = ((step + 1) / total_steps) * 100
        self.status_text.set("Comparison in progress...")

        if step >= total_steps - 1:
            self.root.after(60, self._perform_prediction)
        else:
            self.root.after(100, lambda: self._animate_progress(step + 1))

    def _run_model_explanation(self, model_key: str, image_path: Path) -> dict:
        model_info = self.models[model_key]
        checkpoint = model_info["checkpoint"]
        image_size = resolve_image_size(checkpoint.get("image_size"), checkpoint.get("model_name"))
        return explain_prediction(
            model=model_info["model"],
            image_path=image_path,
            device=self.device,
            image_size=image_size,
            class_names=checkpoint["class_names"],
            model_name=checkpoint.get("model_name"),
            normalization_mean=checkpoint.get("normalization_mean"),
            normalization_std=checkpoint.get("normalization_std"),
            scan_info=self.current_scan_info,
        )

    def _estimate_analysis_accuracy(self, model_key: str, result: dict, scan_info: dict | None = None) -> float | None:
        reference_records = self.models[model_key].get("analysis_reference", [])
        estimated = estimate_prediction_accuracy(
            predicted_class=result["predicted_class"],
            confidence=float(result["confidence"]),
            reference_records=reference_records,
        )
        if estimated is None:
            estimated = self._estimate_accuracy_from_summary(model_key, result)
        if estimated is None:
            return None

        if scan_info is not None and scan_info.get("brain_ct_status") == "uncertain":
            estimated -= 0.02 if result.get("used_inverted_view") else 0.05

        return max(0.0, min(1.0, estimated))

    def _estimate_accuracy_from_summary(self, model_key: str, result: dict) -> float | None:
        summary = self.models[model_key].get("evaluation_summary")
        if not isinstance(summary, dict):
            return None

        metrics = summary.get("test_metrics")
        if not isinstance(metrics, dict):
            metrics = summary.get("zip_reference_test_metrics")
        if not isinstance(metrics, dict):
            return None

        baseline = None
        per_class = metrics.get("per_class")
        if isinstance(per_class, list):
            for class_metrics in per_class:
                if class_metrics.get("class_name") == result["predicted_class"]:
                    precision = class_metrics.get("precision")
                    if precision is not None:
                        baseline = float(precision)
                    break

        if baseline is None:
            accuracy = metrics.get("accuracy", summary.get("best_val_accuracy"))
            if accuracy is None:
                return None
            baseline = float(accuracy)

        confidence = float(result["confidence"])
        return (0.60 * baseline) + (0.40 * confidence)

    def _compose_comparison_summary(self, results: dict[str, dict], scan_info: dict | None = None) -> tuple[str, str, str]:
        pretrained = results["pretrained"]
        custom = results.get("custom")
        pretrained_accuracy = self._estimate_analysis_accuracy("pretrained", pretrained, scan_info)
        pretrained_accuracy_text = (
            f"{pretrained_accuracy * 100:.2f}%" if pretrained_accuracy is not None else "--"
        )

        if custom is None:
            agreement = "Custom checkpoint missing"
            risk_text, _ = _risk_label(pretrained["predicted_class"], float(pretrained["confidence"]))
            summary = (
                f"Pretrained CNN only. Result: {pretrained['predicted_class']}. "
                f"Estimated reliability: {pretrained_accuracy_text}."
            )
            return summary, agreement, risk_text

        same_label = pretrained["predicted_class"] == custom["predicted_class"]
        agreement = "Models agree" if same_label else "Models disagree"
        custom_accuracy = self._estimate_analysis_accuracy("custom", custom, scan_info)
        custom_accuracy_text = f"{custom_accuracy * 100:.2f}%" if custom_accuracy is not None else "--"
        positive_predictions = sum(
            1 for result in (pretrained, custom) if result["predicted_class"] == "hemorrhage"
        )
        if positive_predictions == 2:
            risk_text = "Consensus positive"
        elif positive_predictions == 1:
            risk_text = "Mixed review"
        else:
            risk_text = "Consensus negative"

        summary = (
            f"{agreement}. Risk band: {risk_text}. "
            f"Pretrained reliability {pretrained_accuracy_text}; custom reliability {custom_accuracy_text}."
        )
        return summary, agreement, risk_text

    def _compose_analysis_text(self, results: dict[str, dict], scan_info: dict) -> str:
        pretrained = results["pretrained"]
        custom = results.get("custom")
        pretrained_accuracy = self._estimate_analysis_accuracy("pretrained", pretrained, scan_info)
        custom_accuracy = self._estimate_analysis_accuracy("custom", custom, scan_info) if custom is not None else None

        def _location_sentence(model_name: str, result: dict) -> str:
            if result["localization_status"] == "reliable":
                return f"{model_name} localized the suspicious region in the {result['location_text']}."
            if result["localization_status"] == "tentative":
                return f"{model_name} marked a tentative suspicious region in the {result['location_text']}."
            if result["predicted_class"] == "hemorrhage":
                return f"{model_name} predicted hemorrhage, but the suspicious region could not be localized confidently."
            return f"{model_name} did not mark a suspicious hemorrhage region."

        lines = [
            f"Pretrained CNN: {pretrained['predicted_class']} | confidence {pretrained['confidence'] * 100:.2f}% | estimated reliability {(pretrained_accuracy * 100):.2f}%"
            if pretrained_accuracy is not None
            else f"Pretrained CNN: {pretrained['predicted_class']} | confidence {pretrained['confidence'] * 100:.2f}% | estimated reliability --",
            _location_sentence("Pretrained CNN", pretrained),
        ]
        if custom is not None:
            lines.extend(
                [
                    f"Custom CNN: {custom['predicted_class']} | confidence {custom['confidence'] * 100:.2f}% | estimated reliability {(custom_accuracy * 100):.2f}%"
                    if custom_accuracy is not None
                    else f"Custom CNN: {custom['predicted_class']} | confidence {custom['confidence'] * 100:.2f}% | estimated reliability --",
                    _location_sentence("Custom CNN", custom),
                ]
            )
        lines.append(
            f"Scan quality: {scan_info['quality']}; density pattern: {scan_info['density_pattern']}; framing: {scan_info['framing']}."
        )
        return "\n".join(lines)

    def _apply_model_result(self, model_key: str, result: dict, scan_info: dict | None = None) -> None:
        vars_for_model = self.model_text_vars[model_key]
        detected = result["predicted_class"] == "hemorrhage"
        vars_for_model["result"].set("Hemorrhage Detected" if detected else "No Hemorrhage")
        vars_for_model["confidence"].set(f"{result['confidence'] * 100:.2f}%")
        vars_for_model["hemorrhage_prob"].set(f"{result['probabilities'].get('hemorrhage', 0.0) * 100:.2f}%")
        vars_for_model["no_hemorrhage_prob"].set(
            f"{result['probabilities'].get('no_hemorrhage', 0.0) * 100:.2f}%"
        )
        analysis_accuracy = self._estimate_analysis_accuracy(model_key, result, scan_info)
        vars_for_model["analysis_accuracy"].set(
            f"{analysis_accuracy * 100:.2f}%" if analysis_accuracy is not None else "--"
        )
        if result["localization_status"] == "reliable":
            vars_for_model["region"].set(f"Hemorrhage focus: {result['location_text']}")
        elif result["localization_status"] == "tentative":
            vars_for_model["region"].set(f"Hemorrhage focus: {result['location_text']} (tentative)")
        elif result["predicted_class"] == "hemorrhage":
            vars_for_model["region"].set("Hemorrhage focus: could not be localized clearly")
        else:
            vars_for_model["region"].set("Hemorrhage focus: no suspicious region shown")

        self.model_result_labels[model_key]["result"].configure(
            fg=WARNING if detected else SUCCESS
        )

        if model_key == "pretrained":
            overlay_photo = _make_preview(result["overlay_image"], MODEL_PREVIEW_SIZE)
            self.model_overlay_photos[model_key] = overlay_photo
            self.pretrained_overlay_label.configure(image=overlay_photo, text="")
        else:
            self.model_overlay_photos[model_key] = None

    def _perform_prediction(self) -> None:
        image_path = Path(self.selected_path.get())

        try:
            scan_info = self.current_scan_info or analyze_scan(image_path)
            if scan_info["brain_ct_should_block"]:
                raise ValueError(scan_info["brain_ct_message"])
            results = {"pretrained": self._run_model_explanation("pretrained", image_path)}
            if "custom" in self.models:
                results["custom"] = self._run_model_explanation("custom", image_path)
        except Exception as exc:
            self.is_animating = False
            self.progress.grid_remove()
            self.browse_button.configure(state="normal")
            self.compare_button.configure(state="normal", text="Compare")
            self.clear_button.configure(state="normal")
            messagebox.showerror("Prediction error", str(exc))
            return

        self._apply_model_result("pretrained", results["pretrained"], scan_info)
        if "custom" in results:
            self._apply_model_result("custom", results["custom"], scan_info)

        summary, agreement, risk_text = self._compose_comparison_summary(results, scan_info)
        self.summary_text.set(summary)
        self._update_scan_metrics(scan_info, agreement_text=agreement, risk_text=risk_text)
        self.analysis_text.set(self._compose_analysis_text(results, scan_info))
        self.status_text.set(f"Comparison complete for {image_path.name}")

        self.is_animating = False
        self.progress.grid_remove()
        self.browse_button.configure(state="normal")
        self.compare_button.configure(state="normal", text="Compare")
        self.clear_button.configure(state="normal")


def main() -> None:
    pretrained_candidates = [
        PROJECT_ROOT / "models" / "best_convnext_base_es_uint8.pth",
        PROJECT_ROOT / "models" / "best_convnext_base_es.pth",
        PROJECT_ROOT / "models" / "best_convnext_base.pth",
    ]
    selected_pretrained = None
    for path in pretrained_candidates:
        if not path.exists():
            continue
        model_name = _read_checkpoint_model_name(path)
        if path.name in {"best_convnext_base_es_uint8.pth", "best_convnext_base_es.pth", "best_convnext_base.pth"} and model_name in {"convnext_base", "convnext_medium", "medium_convnext", "base_convnext"}:
            selected_pretrained = path
            break
    if selected_pretrained is None:
        selected_pretrained = PROJECT_ROOT / "models" / "best_convnext_base_es_uint8.pth"

    checkpoint_paths = {
        "pretrained": selected_pretrained,
        "custom": PROJECT_ROOT / "models" / "custom_cnn.pth",
    }

    if not checkpoint_paths["pretrained"].exists():
        raise FileNotFoundError("No pretrained checkpoint found. Train the model before opening the GUI.")
    if not checkpoint_paths["custom"].exists():
        raise FileNotFoundError("No custom checkpoint found at models/custom_cnn.pth.")

    root = tk.Tk()
    PredictionApp(root, checkpoint_paths)
    root.mainloop()


if __name__ == "__main__":
    main()
