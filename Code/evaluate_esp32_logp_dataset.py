"""Evaluate the ESP32 logP model on the full dataset and export paper-ready artifacts.

Example:
    python Code/evaluate_esp32_logp_dataset.py --port COM4

Outputs:
    Code/Results/predictions.csv
    Code/Results/summary_metrics.csv
    Code/Results/worst_cases.csv
    Code/Results/figures/*.pdf

Dependencies:
    pandas
    numpy
    matplotlib
    pyserial
    rdkit-pypi or rdkit
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from serial import SerialException

from send_smiles_to_esp32 import (
    DEFAULT_BAUDRATE,
    DEFAULT_PORT,
    expect_pong,
    request_prediction,
    smiles_to_bitstring,
)


DATA_PATH = Path(__file__).parent / "Dataset" / "250k_rndm_zinc_drugs_clean_3.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "Results"
PLOT_SAMPLE_SIZE = 20_000
WORST_CASE_COUNT = 20
DEFAULT_SAMPLE_SIZE = 1000

PREDICTIONS_HEADER = [
    "sample_index",
    "smiles",
    "actual_logp",
    "predicted_logp",
    "signed_error",
    "absolute_error",
    "squared_error",
    "smape_percent",
    "inference_us",
    "round_trip_ms",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA_PATH, help="CSV file with at least smiles and logP columns")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for raw results, tables, and figures")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port of the ESP32")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baudrate")
    parser.add_argument("--boot-wait", type=float, default=2.0, help="Seconds to wait after opening the serial port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for ESP32 replies")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit applied after subset selection")
    parser.add_argument(
        "--selection-strategy",
        choices=["full", "random", "stratified_logp", "extremes", "latency"],
        default="stratified_logp",
        help="How to choose compounds for evaluation when the full dataset is too slow",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Target subset size for non-full selection strategies",
    )
    parser.add_argument("--skip-run", action="store_true", help="Skip serial evaluation and only rebuild tables/figures from predictions.csv")
    parser.add_argument("--overwrite", action="store_true", help="Delete any existing predictions.csv instead of resuming")
    return parser.parse_args()


def ensure_output_layout(output_dir: Path) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "predictions": output_dir / "predictions.csv",
        "summary_csv": output_dir / "summary_metrics.csv",
        "bands_csv": output_dir / "error_bands.csv",
        "worst_csv": output_dir / "worst_cases.csv",
        "figures_dir": figures_dir,
    }


def load_dataset(dataset_path: Path, max_samples: int | None) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path, usecols=["smiles", "logP"])
    dataset = dataset.rename(columns={"logP": "actual_logp"}).reset_index(names="sample_index")
    dataset["smiles"] = dataset["smiles"].astype(str).str.strip()
    if max_samples is not None:
        dataset = dataset.iloc[:max_samples].copy()
    return dataset


def select_dataset_subset(
    dataset: pd.DataFrame,
    selection_strategy: str,
    sample_size: int,
    max_samples: int | None,
) -> pd.DataFrame:
    if selection_strategy == "full":
        selected = dataset
    elif selection_strategy == "random":
        selected = dataset.sample(n=min(sample_size, len(dataset)), random_state=42).sort_values("sample_index")
    elif selection_strategy == "stratified_logp":
        quantile_bins = min(10, max(2, sample_size // 100))
        ranked = dataset.copy()
        ranked["logp_bin"] = pd.qcut(
            ranked["actual_logp"],
            q=quantile_bins,
            labels=False,
            duplicates="drop",
        )
        per_bin = max(1, sample_size // max(1, ranked["logp_bin"].nunique()))
        selected = ranked.groupby("logp_bin", group_keys=False).apply(
            lambda frame: frame.sample(n=min(per_bin, len(frame)), random_state=42)
        )
        if "logp_bin" in selected.columns:
            selected = selected.drop(columns="logp_bin")
        selected = selected.sort_values("sample_index")
    elif selection_strategy == "extremes":
        if sample_size >= len(dataset):
            selected = dataset
        else:
            edge_count = max(1, sample_size // 4)
            middle_count = max(0, sample_size - 2 * edge_count)
            low = dataset.nsmallest(edge_count, "actual_logp")
            high = dataset.nlargest(edge_count, "actual_logp")
            remaining = dataset.drop(index=low.index.union(high.index))
            middle = remaining.sample(n=min(middle_count, len(remaining)), random_state=42)
            selected = pd.concat([low, middle, high], ignore_index=False).sort_values("sample_index")
    elif selection_strategy == "latency":
        selected = dataset.sample(n=min(sample_size, len(dataset)), random_state=7).sort_values("sample_index")
    else:
        raise ValueError(f"Unsupported selection strategy: {selection_strategy}")

    selected = selected.drop_duplicates(subset="sample_index").copy()
    if max_samples is not None:
        selected = selected.iloc[:max_samples].copy()
    return selected.reset_index(drop=True)


def load_completed_indices(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()

    completed = pd.read_csv(predictions_path, usecols=["sample_index"])
    return set(int(value) for value in completed["sample_index"].dropna().tolist())


def initialize_predictions_file(predictions_path: Path, overwrite: bool) -> None:
    if overwrite and predictions_path.exists():
        predictions_path.unlink()

    if predictions_path.exists():
        return

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTIONS_HEADER)
        writer.writeheader()


def compute_smape_percent(actual_value: float, predicted_value: float) -> float:
    denominator = abs(actual_value) + abs(predicted_value)
    if denominator == 0.0:
        return 0.0
    return 200.0 * abs(predicted_value - actual_value) / denominator


def append_prediction_row(predictions_path: Path, row: dict[str, object]) -> None:
    with predictions_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTIONS_HEADER)
        writer.writerow(row)


def evaluate_dataset(
    dataset: pd.DataFrame,
    predictions_path: Path,
    port: str,
    baudrate: int,
    boot_wait: float,
    timeout: float,
) -> None:
    completed_indices = load_completed_indices(predictions_path)
    pending = dataset.loc[~dataset["sample_index"].isin(completed_indices)]
    if pending.empty:
        print("All requested samples are already present in predictions.csv")
        return

    print(f"Opening {port} at {baudrate} baud for {len(pending)} pending samples...")
    try:
        with serial.Serial(port, baudrate, timeout=0.2, write_timeout=2.0) as connection:
            time.sleep(boot_wait)
            connection.reset_input_buffer()
            connection.reset_output_buffer()
            expect_pong(connection, timeout_seconds=timeout)

            start_time = time.perf_counter()
            for item_number, row in enumerate(pending.itertuples(index=False), start=1):
                error_message = ""
                prediction_value = math.nan
                signed_error = math.nan
                absolute_error = math.nan
                squared_error = math.nan
                smape_percent = math.nan
                inference_us = math.nan
                round_trip_ms = math.nan
                status = "ok"

                try:
                    bitstring = smiles_to_bitstring(row.smiles)
                    request_start = time.perf_counter()
                    prediction_value, inference_us = request_prediction(
                        connection,
                        bitstring,
                        timeout_seconds=timeout,
                    )
                    round_trip_ms = (time.perf_counter() - request_start) * 1000.0
                    signed_error = float(prediction_value - row.actual_logp)
                    absolute_error = abs(signed_error)
                    squared_error = signed_error * signed_error
                    smape_percent = compute_smape_percent(float(row.actual_logp), float(prediction_value))
                except ValueError as exc:
                    status = "invalid_smiles"
                    error_message = str(exc)
                except TimeoutError as exc:
                    status = "timeout"
                    error_message = str(exc)
                except RuntimeError as exc:
                    status = "device_error"
                    error_message = str(exc)

                append_prediction_row(
                    predictions_path,
                    {
                        "sample_index": int(row.sample_index),
                        "smiles": row.smiles,
                        "actual_logp": float(row.actual_logp),
                        "predicted_logp": prediction_value,
                        "signed_error": signed_error,
                        "absolute_error": absolute_error,
                        "squared_error": squared_error,
                        "smape_percent": smape_percent,
                        "inference_us": inference_us,
                        "round_trip_ms": round_trip_ms,
                        "status": status,
                        "error_message": error_message,
                    },
                )

                if item_number == 1 or item_number % 250 == 0:
                    elapsed_seconds = time.perf_counter() - start_time
                    processed_per_second = item_number / elapsed_seconds if elapsed_seconds > 0 else 0.0
                    print(
                        f"Processed {item_number}/{len(pending)} pending samples "
                        f"({processed_per_second:.2f} samples/s)"
                    )
    except SerialException as exc:
        raise RuntimeError(
            f"Could not use {port}: {exc}. Close any serial monitor and rerun with --port {port}."
        ) from exc


def save_summary_tables(valid_rows: pd.DataFrame, all_rows: pd.DataFrame, paths: dict[str, Path]) -> None:
    total_samples = len(all_rows)
    valid_count = len(valid_rows)
    failure_count = total_samples - valid_count

    absolute_errors = valid_rows["absolute_error"].to_numpy(dtype=np.float64)
    squared_errors = valid_rows["squared_error"].to_numpy(dtype=np.float64)
    actual_values = valid_rows["actual_logp"].to_numpy(dtype=np.float64)
    predicted_values = valid_rows["predicted_logp"].to_numpy(dtype=np.float64)
    inference_us = valid_rows["inference_us"].to_numpy(dtype=np.float64)
    round_trip_ms = valid_rows["round_trip_ms"].to_numpy(dtype=np.float64)

    if valid_count == 0:
        summary = pd.DataFrame(
            [{
                "total_samples": total_samples,
                "successful_samples": 0,
                "failed_samples": failure_count,
                "success_rate_percent": 0.0,
            }]
        )
        summary.to_csv(paths["summary_csv"], index=False)
        return

    residual_sum_of_squares = float(np.sum(squared_errors))
    total_sum_of_squares = float(np.sum((actual_values - actual_values.mean()) ** 2))
    r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares if total_sum_of_squares > 0 else math.nan
    pearson_r = float(np.corrcoef(actual_values, predicted_values)[0, 1]) if valid_count > 1 else math.nan

    summary = pd.DataFrame(
        [{
            "total_samples": total_samples,
            "successful_samples": valid_count,
            "failed_samples": failure_count,
            "success_rate_percent": round(100.0 * valid_count / total_samples, 4),
            "mae": round(float(np.mean(absolute_errors)), 6),
            "rmse": round(float(np.sqrt(np.mean(squared_errors))), 6),
            "median_absolute_error": round(float(np.median(absolute_errors)), 6),
            "max_absolute_error": round(float(np.max(absolute_errors)), 6),
            "mean_smape_percent": round(float(valid_rows["smape_percent"].mean()), 6),
            "r_squared": round(r_squared, 6),
            "pearson_r": round(pearson_r, 6),
            "within_0_25_percent": round(float(np.mean(absolute_errors <= 0.25) * 100.0), 4),
            "within_0_50_percent": round(float(np.mean(absolute_errors <= 0.50) * 100.0), 4),
            "within_1_00_percent": round(float(np.mean(absolute_errors <= 1.00) * 100.0), 4),
            "mean_inference_us": round(float(np.mean(inference_us)), 3),
            "p95_inference_us": round(float(np.percentile(inference_us, 95)), 3),
            "mean_round_trip_ms": round(float(np.mean(round_trip_ms)), 3),
            "p95_round_trip_ms": round(float(np.percentile(round_trip_ms, 95)), 3),
        }]
    )
    summary.to_csv(paths["summary_csv"], index=False)

    error_bands = pd.DataFrame(
        [
            {"band": "|error| <= 0.10", "share_percent": round(float(np.mean(absolute_errors <= 0.10) * 100.0), 4)},
            {"band": "|error| <= 0.25", "share_percent": round(float(np.mean(absolute_errors <= 0.25) * 100.0), 4)},
            {"band": "|error| <= 0.50", "share_percent": round(float(np.mean(absolute_errors <= 0.50) * 100.0), 4)},
            {"band": "|error| <= 1.00", "share_percent": round(float(np.mean(absolute_errors <= 1.00) * 100.0), 4)},
            {"band": "|error| > 1.00", "share_percent": round(float(np.mean(absolute_errors > 1.00) * 100.0), 4)},
        ]
    )
    error_bands.to_csv(paths["bands_csv"], index=False)

    worst_cases = valid_rows.nlargest(WORST_CASE_COUNT, "absolute_error")[
        ["sample_index", "smiles", "actual_logp", "predicted_logp", "absolute_error", "inference_us", "round_trip_ms"]
    ].copy()
    worst_cases["smiles"] = worst_cases["smiles"].str.slice(0, 80)
    for column in ["actual_logp", "predicted_logp", "absolute_error", "round_trip_ms"]:
        worst_cases[column] = worst_cases[column].map(lambda value: round(float(value), 6))
    worst_cases["inference_us"] = worst_cases["inference_us"].map(lambda value: round(float(value), 3))
    worst_cases.to_csv(paths["worst_csv"], index=False)


def sample_for_plotting(valid_rows: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(valid_rows) <= sample_size:
        return valid_rows
    return valid_rows.sample(n=sample_size, random_state=42)


def save_figures(valid_rows: pd.DataFrame, figures_dir: Path) -> None:
    if valid_rows.empty:
        return

    plotting_rows = sample_for_plotting(valid_rows, PLOT_SAMPLE_SIZE)

    min_axis_value = float(min(plotting_rows["actual_logp"].min(), plotting_rows["predicted_logp"].min()))
    max_axis_value = float(max(plotting_rows["actual_logp"].max(), plotting_rows["predicted_logp"].max()))

    plt.figure(figsize=(8, 6))
    plt.scatter(plotting_rows["actual_logp"], plotting_rows["predicted_logp"], s=8, alpha=0.25)
    plt.plot([min_axis_value, max_axis_value], [min_axis_value, max_axis_value], linestyle="--", linewidth=1.2)
    plt.xlabel("Reference logP")
    plt.ylabel("ESP32-predicted logP")
    plt.title("ESP32 predictions vs. reference values")
    plt.tight_layout()
    plt.savefig(figures_dir / "scatter_actual_vs_predicted.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(valid_rows["absolute_error"], bins=60, color="#4477AA", edgecolor="white")
    plt.xlabel("Absolute error")
    plt.ylabel("Compound count")
    plt.title("Absolute error distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "absolute_error_histogram.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(plotting_rows["actual_logp"], plotting_rows["signed_error"], s=8, alpha=0.25, color="#CC6677")
    plt.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("Reference logP")
    plt.ylabel("Prediction error")
    plt.title("Residuals vs. reference logP")
    plt.tight_layout()
    plt.savefig(figures_dir / "residuals_vs_actual.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(valid_rows["inference_us"], bins=50, color="#228833", edgecolor="white")
    plt.xlabel("ESP32 inference time [us]")
    plt.ylabel("Compound count")
    plt.title("On-device inference time distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "inference_time_histogram.pdf")
    plt.close()

    cumulative_mae = valid_rows.sort_values("sample_index")["absolute_error"].expanding().mean()
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(cumulative_mae) + 1), cumulative_mae.to_numpy(), color="#AA4499", linewidth=1.2)
    plt.xlabel("Processed compounds")
    plt.ylabel("Cumulative MAE")
    plt.title("Cumulative MAE over evaluation order")
    plt.tight_layout()
    plt.savefig(figures_dir / "cumulative_mae.pdf")
    plt.close()


def build_artifacts(predictions_path: Path, paths: dict[str, Path]) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    all_rows = pd.read_csv(predictions_path)
    valid_rows = all_rows.loc[all_rows["status"] == "ok"].copy()
    save_summary_tables(valid_rows, all_rows, paths)
    save_figures(valid_rows, paths["figures_dir"])
    print(f"Saved tables and figures to {paths['figures_dir'].parent}")


def main() -> int:
    args = parse_args()
    paths = ensure_output_layout(args.output_dir)
    initialize_predictions_file(paths["predictions"], overwrite=args.overwrite)

    if not args.skip_run:
        full_dataset = load_dataset(args.dataset, max_samples=None)
        dataset = select_dataset_subset(
            dataset=full_dataset,
            selection_strategy=args.selection_strategy,
            sample_size=args.sample_size,
            max_samples=args.max_samples,
        )
        print(
            f"Selected {len(dataset)} compounds using strategy={args.selection_strategy} "
            f"from {len(full_dataset)} available rows"
        )
        try:
            evaluate_dataset(
                dataset=dataset,
                predictions_path=paths["predictions"],
                port=args.port,
                baudrate=args.baudrate,
                boot_wait=args.boot_wait,
                timeout=args.timeout,
            )
        except RuntimeError as exc:
            print(exc)
            return 1

    build_artifacts(paths["predictions"], paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())