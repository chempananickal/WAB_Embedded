"""Generate a paper-ready overview figure for the SMILES-to-ESP32 deployment flow.

The figure combines five stages for a single example molecule:
1. Canonicalize the input SMILES.
2. Draw the molecular graph with RDKit.
3. Show how the fingerprint is used during neural-network training.
4. Summarize the LiteRT quantization step.
5. Show how the quantized model is packaged for the ESP32 firmware.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

from ai_edge_litert.interpreter import Interpreter
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import Draw

from mol_preprocessing import canonicalize_smiles, smiles_to_clean_molecule, smiles_to_morgan_fingerprint
from train_logp_tflite import (
    BATCH_SIZE,
    EPOCHS,
    FP_BITS,
    FP_RADIUS,
    MODEL_DIR,
    TEST_FRACTION,
    USE_INT8,
    VALIDATION_FRACTION,
)


DEFAULT_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
OUTPUT_PATH = Path(__file__).parent / "Results" / "figures" / "deploy.pdf"
FIGURE_DPI = 220


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="?", default=DEFAULT_SMILES, help="SMILES string to visualize")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Path of the output PNG figure")
    return parser.parse_args()


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def render_molecule_image(smiles: str) -> Image.Image:
    molecule = smiles_to_clean_molecule(smiles)
    if molecule is None:
        raise ValueError(f"Invalid or uncleanable SMILES: {smiles}")
    return Draw.MolToImage(molecule, size=(700, 420), kekulize=True)


def fingerprint_grid(smiles: str) -> np.ndarray:
    fingerprint = smiles_to_morgan_fingerprint(smiles, radius=FP_RADIUS, n_bits=FP_BITS)
    if fingerprint is None:
        raise ValueError(f"Could not build fingerprint for: {smiles}")
    return fingerprint.reshape(32, 64)


def render_fingerprint_text_image(fingerprint: np.ndarray) -> Image.Image:
    font_path = font_manager.findfont("DejaVu Sans Mono")
    font = ImageFont.truetype(font_path, size=13)
    left, top, right, bottom = font.getbbox("0")
    char_width = right - left
    char_height = bottom - top + 2
    padding_x = 12
    padding_y = 10

    image = Image.new(
        "RGB",
        (padding_x * 2 + char_width * fingerprint.shape[1], padding_y * 2 + char_height * fingerprint.shape[0]),
        color="#f5f9fd",
    )
    draw = ImageDraw.Draw(image)

    for row_index, row in enumerate(fingerprint):
        y = padding_y + row_index * char_height
        for column_index, bit in enumerate(row):
            x = padding_x + column_index * char_width
            character = "1" if int(bit) else "0"
            color = "#0a3f91" if character == "1" else "#8eb8e6"
            draw.text((x, y), character, font=font, fill=color)

    return image


def inspect_quantized_model(model_path: Path) -> dict[str, str]:
    if not model_path.exists():
        return {
            "path": str(model_path),
            "status": "Model file not found",
            "size_kib": "n/a",
            "input": "n/a",
            "output": "n/a",
        }

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = tuple(int(dimension) for dimension in input_details["shape"])
    output_shape = tuple(int(dimension) for dimension in output_details["shape"])
    return {
        "path": str(model_path),
        "status": "Quantized model detected",
        "size_kib": f"{model_path.stat().st_size / 1024:.1f} KiB",
        "input": f"{input_shape}, {np.dtype(input_details['dtype']).name}",
        "output": f"{output_shape}, {np.dtype(output_details['dtype']).name}",
    }


def draw_panel_title(axis: plt.Axes, number: int, title: str, subtitle: str) -> None:
    axis.text(
        0.02,
        0.98,
        f"{number}. {title}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#17324d",
    )
    axis.text(
        0.02,
        0.89,
        subtitle,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#47627a",
    )


def format_fraction(value: float) -> str:
    return f"{value * 100:.0f}%"


def build_figure(smiles: str, output_path: Path) -> Path:
    canonical_smiles = canonicalize_smiles(smiles)
    molecule_image = render_molecule_image(canonical_smiles)
    fingerprint = fingerprint_grid(canonical_smiles)
    fingerprint_image = render_fingerprint_text_image(fingerprint)
    quantized_model = inspect_quantized_model(MODEL_DIR / "logp_model.tflite")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "#eef4f8",
        }
    )

    figure = plt.figure(figsize=(15, 12.2), dpi=FIGURE_DPI)
    grid = GridSpec(3, 2, figure=figure, height_ratios=[0.9, 1.55, 1.55], hspace=0.22, wspace=0.16)

    ax_canonical = figure.add_subplot(grid[0, :])
    ax_molecule = figure.add_subplot(grid[1, 0])
    ax_training = figure.add_subplot(grid[1, 1])
    ax_quant = figure.add_subplot(grid[2, 0])
    ax_flash = figure.add_subplot(grid[2, 1])

    for axis in [ax_canonical, ax_molecule, ax_training, ax_quant, ax_flash]:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        panel = patches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#c8d5df",
            facecolor="white",
            transform=axis.transAxes,
            clip_on=False,
            zorder=-10,
        )
        axis.add_patch(panel)

    figure.suptitle("Pipeline for logP Model Deployment on ESP32", fontsize=20, fontweight="bold", y=0.985)
    figure.text(
        0.5,
        0.955,
        "Aspirin is shown as an example input; the neural network is trained on the training subset of the dataset.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4f6272",
    )

    draw_panel_title(ax_canonical, 1, "Canonicalize the Input", "RDKit standardizes the example before feature generation.")
    ax_canonical.text(0.03, 0.60, "Input SMILES", fontsize=11, fontweight="bold", color="#17324d", transform=ax_canonical.transAxes)
    ax_canonical.text(0.03, 0.45, wrap(smiles, 60), fontsize=12.5, color="#243746", transform=ax_canonical.transAxes)
    ax_canonical.annotate(
        "",
        xy=(0.59, 0.52),
        xytext=(0.35, 0.52),
        arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": "#2b7a78"},
        xycoords=ax_canonical.transAxes,
    )
    ax_canonical.text(0.44, 0.61, "RDKit\ncleanup +\ncanonicalization", fontsize=10, color="#2b7a78", ha="center", transform=ax_canonical.transAxes)
    ax_canonical.text(0.62, 0.60, "Canonical SMILES", fontsize=11, fontweight="bold", color="#17324d", transform=ax_canonical.transAxes)
    ax_canonical.text(0.62, 0.45, wrap(canonical_smiles, 40), fontsize=12.5, color="#243746", transform=ax_canonical.transAxes)

    draw_panel_title(ax_molecule, 2, "Build the Molecule Graph", "Canonical SMILES are parsed into an RDKit molecule and then fingerprinted with Morgan/ECFP4.")
    smiles_box = patches.FancyBboxPatch(
        (0.06, 0.66),
        0.36,
        0.12,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#f5f8fb",
        edgecolor="#d7e3ee",
        linewidth=1.1,
        transform=ax_molecule.transAxes,
    )
    ax_molecule.add_patch(smiles_box)
    ax_molecule.text(0.08, 0.73, "Canonical SMILES", fontsize=9.2, fontweight="bold", color="#17324d", transform=ax_molecule.transAxes)
    ax_molecule.text(0.08, 0.68, canonical_smiles, fontsize=9.5, color="#243746", transform=ax_molecule.transAxes)
    ax_molecule.annotate(
        "",
        xy=(0.22, 0.62),
        xytext=(0.24, 0.66),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#6c8ca3"},
        xycoords=ax_molecule.transAxes,
    )
    molecule_inset = ax_molecule.inset_axes([0.05, 0.16, 0.48, 0.48])
    molecule_inset.imshow(molecule_image)
    molecule_inset.set_xticks([])
    molecule_inset.set_yticks([])
    molecule_inset.set_frame_on(False)
    ax_molecule.annotate(
        "",
        xy=(0.71, 0.46),
        xytext=(0.56, 0.46),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#2b7a78"},
        xycoords=ax_molecule.transAxes,
    )
    ax_molecule.text(0.635, 0.61, "Morgan\nradius = 2\n2048 bits", ha="center", va="center", fontsize=9.5, color="#2b7a78", transform=ax_molecule.transAxes)
    molecule_fp_inset = ax_molecule.inset_axes([0.72, 0.18, 0.22, 0.56])
    molecule_fp_inset.imshow(fingerprint_image)
    molecule_fp_inset.set_xticks([])
    molecule_fp_inset.set_yticks([])
    molecule_fp_inset.set_frame_on(True)
    for spine in molecule_fp_inset.spines.values():
        spine.set_edgecolor("#d7e3ee")
    ax_molecule.text(
        0.03,
        0.06,
        "Canonical SMILES -> RDKit Mol object -> Morgan fingerprint.",
        fontsize=8.7,
        color="#4f6272",
        transform=ax_molecule.transAxes,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f5f8fb", "edgecolor": "#dce6ee"},
    )

    draw_panel_title(ax_training, 3, "Train the Neural Network", "Morgan fingerprints become the model input during supervised training.")
    ax_training.text(0.27, 0.75, "Aspirin ECFP4 fingerprint\n(32 lines × 64 bits)", ha="center", va="center", fontsize=10, color="#243746", transform=ax_training.transAxes)
    inset = ax_training.inset_axes([0.05, 0.20, 0.48, 0.54])
    inset.imshow(fingerprint_image)
    inset.set_xticks([])
    inset.set_yticks([])

    layer_centers = [0.64, 0.78, 0.92]
    layer_width = 0.074
    layer_height = 0.18
    layer_y = 0.385
    layer_mid_y = layer_y + layer_height / 2
    layer_labels = [f"Input\n{FP_BITS}", "Dense\n128 + ReLU", "Output\n1 logP"]
    layer_colors = ["#d9edf7", "#fde2b8", "#d6eadf"]
    for center_x, label, color in zip(layer_centers, layer_labels, layer_colors):
        rect = patches.FancyBboxPatch(
            (center_x - layer_width / 2, layer_y),
            layer_width,
            layer_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color,
            edgecolor="#9db4c5",
            linewidth=1.3,
            transform=ax_training.transAxes,
        )
        ax_training.add_patch(rect)
        ax_training.text(center_x, layer_mid_y, label, ha="center", va="center", fontsize=7.7, color="#243746", transform=ax_training.transAxes)
    ax_training.annotate(
        "",
        xy=(layer_centers[0] - layer_width / 2, layer_mid_y),
        xytext=(0.54, layer_mid_y),
        arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6c8ca3"},
        xycoords=ax_training.transAxes,
    )
    ax_training.annotate(
        "",
        xy=(layer_centers[1] - layer_width / 2, layer_mid_y),
        xytext=(layer_centers[0] + layer_width / 2, layer_mid_y),
        arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6c8ca3"},
        xycoords=ax_training.transAxes,
    )
    ax_training.annotate(
        "",
        xy=(layer_centers[2] - layer_width / 2, layer_mid_y),
        xytext=(layer_centers[1] + layer_width / 2, layer_mid_y),
        arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#6c8ca3"},
        xycoords=ax_training.transAxes,
    )
    training_text = (
        f"Dataset split: train {100 - int((TEST_FRACTION + VALIDATION_FRACTION) * 100)}% | "
        f"validation {format_fraction(VALIDATION_FRACTION)} | test {format_fraction(TEST_FRACTION)}\n"
        f"Hyperparameters: batch size {BATCH_SIZE}, up to {EPOCHS} epochs, Adam optimizer, MSE loss"
    )
    ax_training.text(0.03, 0.08, training_text, fontsize=9.2, color="#4f6272", transform=ax_training.transAxes)

    draw_panel_title(ax_quant, 4, "Quantize the Model", "Post-training LiteRT conversion compresses the model for edge deployment.")
    quant_box = patches.FancyBboxPatch((0.05, 0.20), 0.28, 0.48, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#f8f4e8", edgecolor="#d8c48f", linewidth=1.2, transform=ax_quant.transAxes)
    ax_quant.add_patch(quant_box)
    ax_quant.text(0.19, 0.44, "Keras\nmodel", ha="center", va="center", fontsize=14, fontweight="bold", color="#735c0f", transform=ax_quant.transAxes)
    ax_quant.annotate("", xy=(0.55, 0.44), xytext=(0.34, 0.44), arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": "#2b7a78"}, xycoords=ax_quant.transAxes)
    ax_quant.text(0.445, 0.54, "LiteRT\nconverter", ha="center", va="center", fontsize=10, color="#2b7a78", transform=ax_quant.transAxes)
    quant_target = patches.FancyBboxPatch((0.58, 0.20), 0.30, 0.48, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#edf6ed", edgecolor="#9dc4a8", linewidth=1.2, transform=ax_quant.transAxes)
    ax_quant.add_patch(quant_target)
    ax_quant.text(0.73, 0.49, "Quantized\n.tflite", ha="center", va="center", fontsize=14, fontweight="bold", color="#1d5f35", transform=ax_quant.transAxes)
    summary_cards = [
        (0.05, 0.055, 0.22, 0.115, "Quantization", f"PTQ {('int8' if USE_INT8 else 'float')}", 0.078, 0.026, 8.6),
        (0.31, 0.055, 0.20, 0.115, "Model size", quantized_model["size_kib"], 0.078, 0.026, 8.6),
        (0.55, 0.045, 0.33, 0.125, "Tensor types", f"in: {quantized_model['input']}\nout: {quantized_model['output']}", 0.093, 0.012, 7.0),
    ]
    for x, y, width, height, title, value, title_offset, value_offset, value_fontsize in summary_cards:
        card = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor="#f5f8fb",
            edgecolor="#d7e3ee",
            linewidth=1.0,
            transform=ax_quant.transAxes,
        )
        ax_quant.add_patch(card)
        ax_quant.text(x + 0.015, y + title_offset, title, fontsize=8.0, fontweight="bold", color="#6b8297", transform=ax_quant.transAxes)
        ax_quant.text(x + 0.015, y + value_offset, value, fontsize=value_fontsize, color="#243746", transform=ax_quant.transAxes)

    draw_panel_title(ax_flash, 5, "Flash to the ESP32", "The quantized model is converted into firmware assets and flashed with ESP-IDF.")
    packaging_steps = [
        (0.07, 0.42, 0.14, 0.16, ".tflite", "export"),
        (0.25, 0.42, 0.14, 0.16, "C array", "convert"),
        (0.43, 0.42, 0.14, 0.16, "ESP-IDF", "build"),
    ]
    for x, y, width, height, title, caption in packaging_steps:
        step_box = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#f5f8fb",
            edgecolor="#c9d6e2",
            linewidth=1.2,
            transform=ax_flash.transAxes,
        )
        ax_flash.add_patch(step_box)
        ax_flash.text(x + width / 2, y + 0.10, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#17324d", transform=ax_flash.transAxes)
        ax_flash.text(x + width / 2, y + 0.04, caption, ha="center", va="center", fontsize=8.5, color="#6b8297", transform=ax_flash.transAxes)
    ax_flash.annotate("", xy=(0.25, 0.50), xytext=(0.21, 0.50), arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#d1495b"}, xycoords=ax_flash.transAxes)
    ax_flash.annotate("", xy=(0.43, 0.50), xytext=(0.39, 0.50), arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#d1495b"}, xycoords=ax_flash.transAxes)
    ax_flash.annotate("", xy=(0.69, 0.50), xytext=(0.57, 0.50), arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": "#d1495b"}, xycoords=ax_flash.transAxes)
    ax_flash.text(0.63, 0.58, "flash", ha="center", va="center", fontsize=9, color="#d1495b", transform=ax_flash.transAxes)
    board = patches.FancyBboxPatch((0.70, 0.20), 0.20, 0.52, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#17324d", edgecolor="#0e2032", linewidth=1.2, transform=ax_flash.transAxes)
    ax_flash.add_patch(board)
    for x in np.linspace(0.71, 0.89, 7):
        ax_flash.add_line(plt.Line2D([x, x], [0.16, 0.20], transform=ax_flash.transAxes, color="#94a9ba", linewidth=2))
        ax_flash.add_line(plt.Line2D([x, x], [0.72, 0.76], transform=ax_flash.transAxes, color="#94a9ba", linewidth=2))
    chip = patches.Rectangle((0.75, 0.34), 0.10, 0.22, facecolor="#243746", edgecolor="#8bb3c7", linewidth=1.2, transform=ax_flash.transAxes)
    ax_flash.add_patch(chip)
    ax_flash.text(0.80, 0.59, "ESP32", ha="center", va="center", fontsize=16, fontweight="bold", color="white", transform=ax_flash.transAxes)
    ax_flash.text(0.80, 0.45, "TFLite Micro\nfirmware", ha="center", va="center", fontsize=10, color="#d8e7f0", transform=ax_flash.transAxes)
    ax_flash.text(0.09, 0.12, "USB/UART flashing via the ESP-IDF VS Code Extension", fontsize=9.2, color="#4f6272", transform=ax_flash.transAxes)

    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(figure)
    return pdf_path


def main() -> int:
    args = parse_args()
    output_path = build_figure(args.smiles, args.output)
    print(f"Saved deployment figure to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())