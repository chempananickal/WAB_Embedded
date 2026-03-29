# Code Folder README

This folder contains the full workflow for the logP model:

1. Train and export a quantized LiteRT/TFLite model on the PC.
2. Flash an ESP-IDF firmware project that runs that model on the ESP32.
3. Use a Python script on the PC to turn SMILES into Morgan fingerprints.
4. Send those fingerprints to the ESP32 over serial.
5. Read the predicted logP back from the ESP32.

This README describes the real workflow used in this repository.
The old Arduino-based instructions are no longer relevant.

## What matters most

- ESP32 firmware project: `Code/esp32_logp`
- Host sender script: `Code/send_smiles_to_esp32.py`
- Training/export script: `Code/train_logp_tflite.py`
- Model blob used by firmware: `Code/artifacts/logp_model_tflite.cc`
- Model header used by firmware: `Code/artifacts/logp_model_tflite.h`
- Dataset evaluation pipeline: `Code/evaluate_esp32_logp_dataset.py`

## 1) Python environment on the PC

Use the same environment for training, sending SMILES to the ESP32, and running the evaluation pipeline.

Install the required Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install rdkit pandas numpy scikit-learn tensorflow ai-edge-litert pyserial matplotlib
```

If you use a dedicated environment, activate it before running any of the commands below.

## 2) Train and export the model

Run:

```bash
python Code/train_logp_tflite.py
```

What this does:

- Loads the SMILES/logP dataset.
- Converts SMILES to 2048-bit Morgan fingerprints.
- Trains a small Keras regression model.
- Exports a fully int8-quantized `.tflite` model.

Main output:

- `Code/artifacts/logp_model.tflite`

Important:

- The script is currently configured for `USE_INT8 = True`.
- That is required for the ESP32 firmware in this repo. Without quantization, the model will not fit in the ESP32 memory.

## 3) Convert the model to C source

If you retrain the model, regenerate the C array used by the firmware:

```bash
python Code/convert_tflite_to_c.py
```

This updates:

- `Code/artifacts/logp_model_tflite.cc`
- `Code/artifacts/logp_model_tflite.h`

If you do not retrain the model, you can keep the existing generated files.

## 4) Build the ESP-IDF firmware

The actual ESP32 project is:

- `Code/esp32_logp`

It uses Espressif's `esp-tflite-micro` component through `idf_component.yml`.

From the workspace root:

```bash
cd Code/esp32_logp
idf.py set-target esp32
idf.py build
```

The first build will download the managed dependency automatically.

## 5) Flash the ESP32 on COM4

Flash the firmware to the board:

```bash
cd Code/esp32_logp
idf.py -p COM4 flash
```

Optional: watch boot output once to confirm the firmware started correctly:

```bash
idf.py -p COM4 monitor
```

You should see a line that starts with `READY`.

Important:

- Close the serial monitor before using `send_smiles_to_esp32.py`.
- Only one program can own `COM4` at a time.

## 6) Run one prediction from the PC

After the firmware is flashed and running, go back to the workspace root and run:

```bash
python Code/send_smiles_to_esp32.py "CCO"
```

Or interactive mode:

```bash
python Code/send_smiles_to_esp32.py
```

What the script does:

- Canonicalizes and cleans the SMILES.
- Computes a 2048-bit Morgan fingerprint on the PC.
- Opens `COM4`.
- Sends the fingerprint to the ESP32.
- Waits for the model result.
- Prints the predicted logP and ESP32 inference time.

Example output:

```text
Connecting to COM4 at 115200 baud...
SMILES: CCO
Canonical cleaned SMILES: CCO
Predicted logP: 0.123456
ESP32 inference time: 18432 us
```

## 7) Serial protocol used by the firmware

The ESP32 firmware accepts simple line-based commands:

- `PING`
- `HELP`
- `FP <2048 bits of 0 or 1>`

The firmware replies with one of:

- `PONG`
- `INFO ...`
- `RESULT <predicted_logP> <inference_time_us>`
- `ERROR ...`

You normally do not need to type these by hand because the Python script handles them.

## 8) Quantization details

The deployed model uses int8 I/O.

Input mapping on the ESP32:

- fingerprint bit `0` becomes `-128`
- fingerprint bit `1` becomes `127`

The model metadata currently used by the firmware is:

- input shape: `[1, 2048]`
- input dtype: `int8`
- output shape: `[1, 1]`
- output dtype: `int8`

## 9) Evaluate the ESP32 across the dataset

To benchmark the flashed ESP32 against the dataset and generate CSV tables and figures, run:

```bash
python Code/evaluate_esp32_logp_dataset.py --port COM4
```

Default behavior:

- Uses `Code/artifacts/test_set.csv` if it exists
- Falls back to `Code/Dataset/250k_rndm_zinc_drugs_clean_3.csv` if no exported test split is available
- Selects a representative subset by default
- Sends each fingerprint to the ESP32 over serial
- Stores predictions and metrics in `Code/Results`
- Exports PDF plots for the report

Main outputs:

- `Code/Results/predictions.csv`
- `Code/Results/summary_metrics.csv`
- `Code/Results/error_bands.csv`
- `Code/Results/worst_cases.csv`
- `Code/Results/figures/*.pdf`

Useful examples:

```bash
python Code/evaluate_esp32_logp_dataset.py --port COM4 --selection-strategy stratified_logp --sample-size 1000
python Code/evaluate_esp32_logp_dataset.py --port COM4 --selection-strategy extremes --sample-size 400
python Code/evaluate_esp32_logp_dataset.py --port COM4 --selection-strategy random --sample-size 1000
python Code/evaluate_esp32_logp_dataset.py --port COM4 --selection-strategy full
python Code/evaluate_esp32_logp_dataset.py --port COM4 --dataset Code/artifacts/test_set.csv
python Code/evaluate_esp32_logp_dataset.py --port COM4 --dataset Code/Dataset/250k_rndm_zinc_drugs_clean_3.csv
python Code/evaluate_esp32_logp_dataset.py --port COM4 --skip-run
```

## 10) Common problems

### `Access is denied` on `COM4`

Another program already owns the serial port.

Close things like:

- `idf.py monitor`
- VS Code serial monitor
- Arduino Serial Monitor
- PuTTY
- another Python script using `COM4`

Then rerun:

```bash
python Code/send_smiles_to_esp32.py "CCO"
```

### `ERROR allocate_tensors_failed`

Increase `kTensorArenaSize` in `Code/esp32_logp/main/main.cc` and rebuild.

### No response after opening the serial port

Opening the port may reset the board. Wait a couple of seconds and try again. The sender script already does this automatically.

### Invalid SMILES

The sender script rejects invalid or uncleanable SMILES before anything is sent to the ESP32.

## 11) File map

- `Code/train_logp_tflite.py`: train and export the quantized model
- `Code/convert_tflite_to_c.py`: convert `.tflite` model to C source/header
- `Code/send_smiles_to_esp32.py`: send one SMILES to the ESP32 and print the result
- `Code/evaluate_esp32_logp_dataset.py`: run many samples through the ESP32 and export metrics/plots
- `Code/esp32_logp/`: ESP-IDF firmware project
- `Code/artifacts/`: exported model files used by the firmware
- `Code/Dataset/250k_rndm_zinc_drugs_clean_3.csv`: source dataset
