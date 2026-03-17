"""Train a minimal logP regressor and export a quantized model for LiteRT.

Dependencies (pip): rdkit-pypi, tensorflow, pandas, numpy, scikit-learn,
ai-edge-litert
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
import tensorflow as tf


DATA_PATH = Path(__file__).parent / "Dataset" / "250k_rndm_zinc_drugs_clean_3.csv"
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
BATCH_SIZE = 256
EPOCHS = 20
TEST_FRACTION = 0.1
USE_INT8 = True  # Set True for full int8 quantization (recommended for ESP32)
VALIDATION_SPLIT = 0.1



def smiles_to_morgan(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol).astype(np.uint8)


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    fps = df["smiles"].map(lambda s: smiles_to_morgan(s, FP_RADIUS, FP_BITS))
    mask = fps.notna()
    x = np.stack(fps[mask].to_numpy()).astype(np.float32)
    y = df.loc[mask, "logP"].to_numpy(dtype=np.float32)
    return x, y


def build_model(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def representative_dataset(x_samples: np.ndarray) -> Iterable[list[np.ndarray]]:
    for i in range(min(len(x_samples), 500)):
        yield [x_samples[i : i + 1]]


def convert_to_litert_model(model: tf.keras.Model, x_train: np.ndarray) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if USE_INT8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(x_train)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def evaluate_litert_model(
    model_path: Path, x_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    predictions = np.empty(len(x_test), dtype=np.float32)

    start_time = time.perf_counter()
    for index in range(len(x_test)):
        sample = x_test[index : index + 1]
        if input_details["dtype"] == np.int8:
            scale, zero_point = input_details["quantization"]
            sample = np.round(sample / scale + zero_point).astype(np.int8)
        else:
            sample = sample.astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        if output_details["dtype"] == np.int8:
            scale, zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        predictions[index] = float(output.squeeze())

    elapsed = time.perf_counter() - start_time
    mae = float(np.mean(np.abs(y_test - predictions)))
    milliseconds_per_sample = float((elapsed / len(x_test)) * 1000.0)
    return mae, milliseconds_per_sample


def main() -> None:
    x, y = load_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )

    model = build_model(FP_BITS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)

    litert_model = convert_to_litert_model(model, x_train)
    model_path = MODEL_DIR / "logp_model.tflite"
    model_path.write_bytes(litert_model)

    litert_mae, litert_ms = evaluate_litert_model(model_path, x_test, y_test)

    rng = np.random.default_rng(RANDOM_SEED)
    subset_idx = rng.choice(len(x_train), size=min(20_000, len(x_train)), replace=False)
    x_small, y_small = x_train[subset_idx], y_train[subset_idx]
    classical_models = [
        ("Ridge", Ridge(alpha=1.0)),
        ("LinearSVR", LinearSVR(C=1.0, epsilon=0.1, dual="auto", max_iter=5000, random_state=RANDOM_SEED)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
    ]

    print(f"Saved LiteRT model to: {model_path}")
    print(f"Keras test MAE: {test_mae:.4f}")
    print(f"LiteRT test MAE: {litert_mae:.4f}")
    print(f"LiteRT inference time: {litert_ms:.3f} ms/sample")
    print("Classical model comparison:")
    for name, estimator in classical_models:
        estimator.fit(x_small, y_small)
        mae = float(np.mean(np.abs(y_test - estimator.predict(x_test))))
        print(f"  {name}: MAE={mae:.4f} (train n={len(x_small)}, test n={len(x_test)})")


if __name__ == "__main__":
    main()
