"""Train a minimal logP regressor and export a quantized model for LiteRT.

Dependencies (pip): rdkit-pypi, tensorflow, pandas, numpy, scikit-learn,
ai-edge-litert
"""
from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Iterable

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Must be set before TF is imported; suppresses pre-absl log noise
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from mol_preprocessing import smiles_to_morgan_fingerprint



DATA_PATH = Path(__file__).parent / "Dataset" / "250k_rndm_zinc_drugs_clean_3.csv"
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
BATCH_SIZE = 256
EPOCHS = 20
TEST_FRACTION = 0.1
USE_INT8 = True  # Full int8 quantization, required for ESP32
VALIDATION_FRACTION = 0.1


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SMILES/logP pairs from CSV and return Morgan fingerprint matrix and labels."""
    df = pd.read_csv(csv_path)
    fps = df["smiles"].map(lambda s: smiles_to_morgan_fingerprint(s, FP_RADIUS, FP_BITS))
    valid = fps.notna()
    x = np.stack(fps[valid].to_numpy()).astype(np.float32)
    y = df.loc[valid, "logP"].to_numpy(dtype=np.float32)
    return x, y


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


def representative_dataset(x_train: np.ndarray) -> Iterable[list[np.ndarray]]:
    """Yield 500 calibration samples for post-training quantization."""
    for i in range(min(len(x_train), 500)):
        yield [x_train[i : i + 1]]


def convert_to_litert(model: tf.keras.Model, x_train: np.ndarray) -> bytes:
    """Convert a Keras model to a (optionally int8-quantized) TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if USE_INT8:
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            lambda: representative_dataset(x_train)
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


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


def export_model(model: tf.keras.Model, x_train: np.ndarray) -> Path:
    """Serialize the quantized TFLite model to disk and return its path."""
    model_path = MODEL_DIR / "logp_model.tflite"
    model_path.write_bytes(convert_to_litert(model, x_train))
    return model_path


def print_results(
    model_path: Path,
    keras_mae: float,
    litert_metrics: dict[str, float],
    litert_ms: float,
) -> None:
    """Print a human-readable summary of training and LiteRT evaluation results."""
    m = litert_metrics
    print(f"\nSaved model to: {model_path}")
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
    x, y = load_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )

    model = train_model(build_model(FP_BITS), x_train, y_train)
    _, keras_mae = model.evaluate(x_test, y_test, verbose=0)

    model_path = export_model(model, x_train)
    litert_preds, litert_ms = evaluate_litert(model_path, x_test)
    litert_metrics = compute_metrics(y_test, litert_preds)

    print_results(model_path, keras_mae, litert_metrics, litert_ms)


if __name__ == "__main__":
    main()
