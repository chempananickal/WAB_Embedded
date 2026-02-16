"""Train a simple logP regressor from SMILES and export to TFLite.

Dependencies (pip): rdkit-pypi, tensorflow, pandas, numpy, scikit-learn
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import tensorflow as tf


DATA_PATH = Path(__file__).parent / "Dataset" / "250k_rndm_zinc_drugs_clean_3.csv"
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
FP_BITS = 2048
FP_RADIUS = 2
BATCH_SIZE = 256
EPOCHS = 20
USE_INT8 = False  # Set True for full int8 quantization (recommended for ESP32)


def smiles_to_morgan(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    features = []
    targets = []
    for _, row in df.iterrows():
        fp = smiles_to_morgan(row["smiles"], FP_RADIUS, FP_BITS)
        if fp is None:
            continue
        features.append(fp)
        targets.append(row["logP"])

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return x, y


def build_model(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def representative_dataset(x_samples: np.ndarray) -> Iterable[list[np.ndarray]]:
    for i in range(min(len(x_samples), 500)):
        yield [x_samples[i : i + 1]]


def main() -> None:
    x, y = load_dataset(DATA_PATH)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=RANDOM_SEED
    )

    model = build_model(FP_BITS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    saved_model_dir = MODEL_DIR / "logp_model"
    model.save(saved_model_dir, include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
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

    tflite_model = converter.convert()
    tflite_path = MODEL_DIR / "logp_model.tflite"
    tflite_path.write_bytes(tflite_model)

    print(f"Saved TFLite model to: {tflite_path}")


if __name__ == "__main__":
    main()
