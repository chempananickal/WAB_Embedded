"""Train a logP predictor and export a quantized model for LiteRT.

Dependencies (pip): rdkit, tensorflow, pandas, numpy, scikit-learn, ai-edge-litert
"""
from __future__ import annotations

from pathlib import Path
import csv
import os
import random
import time
from typing import Iterable

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Must be set before TF is imported; suppresses pre-absl log noise
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from mol_preprocessing import smiles_to_morgan_fingerprint


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "Dataset"
DATA_PATH = DATA_DIR / "250k_rndm_zinc_drugs_clean_3.csv"
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
BATCH_SIZE = 256
EPOCHS = 20
TEST_FRACTION = 0.1
VALIDATION_FRACTION = 0.1
USE_INT8 = True  # Full int8 quantization, required for ESP32


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow so training is fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load valid SMILES/logP rows and return the rows, fingerprints, and labels."""
    df = pd.read_csv(csv_path)
    cleaned_smiles = df["smiles"].astype(str).str.strip()
    fps = cleaned_smiles.map(lambda s: smiles_to_morgan_fingerprint(s, FP_RADIUS, FP_BITS))
    valid = fps.notna()
    valid_df = df.loc[valid].copy().reset_index(names="source_index")
    valid_df["smiles"] = cleaned_smiles.loc[valid].to_numpy()
    x = np.stack(fps[valid].to_numpy()).astype(np.float32)
    y = valid_df["logP"].to_numpy(dtype=np.float32)
    return valid_df, x, y


def export_split_csvs(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> tuple[Path, Path]:
    """Persist the exact train/test split used for model development."""
    train_path = DATA_DIR / "training_set.csv"
    test_path = DATA_DIR / "test_set.csv"
    train_rows.to_csv(train_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    test_rows.to_csv(test_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return train_path, test_path


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_dim: int) -> tf.keras.Model:
    """Return a two-layer DNN that maps a fingerprint vector to a single logP value."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])


def train_model(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:
    """Compile and fit the model; returns the model with best validation weights."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    model.fit(
        x_train, y_train,
        validation_split=VALIDATION_FRACTION,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=2,
    )
    return model


# ---------------------------------------------------------------------------
# Quantization and export
# ---------------------------------------------------------------------------

def representative_dataset(x_train: np.ndarray, y_train: np.ndarray) -> Iterable[list[np.ndarray]]:
    """Yield calibration samples that span the full logP range for accurate output quantization.

    Naive sequential sampling risks missing extreme logP values, which causes the output
    quantization scale to be calibrated to a narrow range and clamps predictions at the ceiling.
    Sorting by logP and sampling evenly across the sorted order ensures the calibration set
    covers the full output distribution.
    """
    n = min(len(x_train), 500)
    indices = np.argsort(y_train)
    step = max(1, len(indices) // n)
    calibration_indices = indices[::step][:n]
    for i in calibration_indices:
        yield [x_train[i : i + 1]]


def convert_to_litert(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray) -> bytes:
    """Convert a Keras model to a (optionally int8-quantized) TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if USE_INT8:
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            lambda: representative_dataset(x_train, y_train)
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32  # float32 output avoids int8 clamping on the output tensor
    return converter.convert()


def export_model(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray) -> Path:
    """Serialize the quantized TFLite model to disk and return its path."""
    model_path = MODEL_DIR / "logp_model.tflite"
    model_path.write_bytes(convert_to_litert(model, x_train, y_train))
    return model_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _dequantize(tensor: np.ndarray, details: dict) -> np.ndarray:
    """Undo int8 quantization using the scale and zero-point stored in tensor details."""
    scale, zero_point = details["quantization"]
    return (tensor.astype(np.float32) - zero_point) * scale


def evaluate_litert(
    model_path: Path, x_test: np.ndarray
) -> tuple[np.ndarray, float]:
    """Run inference via LiteRT; return predictions array and average milliseconds per sample."""
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]

    predictions = np.empty(len(x_test), dtype=np.float32)
    start = time.perf_counter()

    for i, sample in enumerate(x_test):
        sample = sample[np.newaxis]
        if input_det["dtype"] == np.int8:
            scale, zero_point = input_det["quantization"]
            sample = np.round(sample / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_det["index"], sample)
        interpreter.invoke()

        output = interpreter.get_tensor(output_det["index"])
        if output_det["dtype"] == np.int8:
            output = _dequantize(output, output_det)
        predictions[i] = float(output.squeeze())

    ms_per_sample = (time.perf_counter() - start) / len(x_test) * 1000
    return predictions, ms_per_sample


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics comparing predicted and true logP values."""
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    within_half = float(np.mean(np.abs(errors) <= 0.5) * 100)
    max_err = float(np.max(np.abs(errors)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "within_half": within_half, "max_err": max_err}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_results(
    model_path: Path,
    train_csv_path: Path,
    test_csv_path: Path,
    keras_mae: float,
    litert_metrics: dict[str, float],
    litert_ms: float,
) -> None:
    """Print a human-readable summary of training and LiteRT evaluation results."""
    m = litert_metrics
    print(f"\nSaved model to: {model_path}")
    print(f"Saved training split to: {train_csv_path}")
    print(f"Saved test split to: {test_csv_path}")
    print(f"\n--- Accuracy (on {int(100 * TEST_FRACTION)}% held-out test set) ---")
    print(f"  Average error (MAE):          {m['mae']:.3f} logP units  (Keras: {keras_mae:.3f})")
    print(f"  Typical error (RMSE):         {m['rmse']:.3f} logP units")
    print(f"  Variance (R²):                {m['r2']:.4f}  (1.0 = perfect)")
    print(f"  Within ±0.5 logP units:       {m['within_half']:.1f}%")
    print(f"  Worst-case error:             {m['max_err']:.3f} logP units")
    print(f"\n--- Speed ---")
    print(f"  LiteRT inference time:        {litert_ms:.3f} ms/sample")


def main() -> None:
    """Train a logP DNN, export it as a quantized TFLite model, and report accuracy."""
    _seed_everything(RANDOM_SEED)
    rows, x, y = load_dataset(DATA_PATH)
    train_rows, test_rows, x_train, x_test, y_train, y_test = train_test_split(
        rows, x, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )

    model = train_model(build_model(FP_BITS), x_train, y_train)
    _, keras_mae = model.evaluate(x_test, y_test, verbose=0)

    train_csv_path, test_csv_path = export_split_csvs(train_rows, test_rows)
    model_path = export_model(model, x_train, y_train)
    litert_preds, litert_ms = evaluate_litert(model_path, x_test)
    litert_metrics = compute_metrics(y_test, litert_preds)

    print_results(model_path, train_csv_path, test_csv_path, keras_mae, litert_metrics, litert_ms)


if __name__ == "__main__":
    main()
