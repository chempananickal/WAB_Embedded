# Code Folder README (Start Here)

This folder contains the training code and example C++ files for running a logP model with TensorFlow Lite (TFLite). The steps below assume you are a total beginner and are using Windows.

## 1) Install Python

1. Download Python 3.10 or newer from <https://www.python.org/downloads/> .
2. During installation, check the box "Add Python to PATH".
3. Open a new PowerShell window and verify:
   - Run: `python --version`

## 2) Create a virtual environment (recommended)

1. In PowerShell, go to this folder:
   - `cd ~\WAB_Embedded\Code`
2. Create a virtual environment:
   - `python -m venv .venv`
3. Activate it:
   - `./.venv/Scripts/Activate.ps1`

If PowerShell blocks scripts, run once:

- `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 3) Install Python dependencies

Run:

- `python -m pip install --upgrade pip`
- `python -m pip install rdkit-pypi tensorflow pandas numpy scikit-learn`

## 4) Train and export the TFLite model

Run:

- `python train_logp_tflite.py`

Outputs:

- Model folder: `Code/artifacts/logp_model/`
- TFLite file: `Code/artifacts/logp_model.tflite`

Notes:

- To enable full int8 quantization (recommended for ESP32), edit `train_logp_tflite.py` and set `USE_INT8 = True`, then re-run the script.

## 5) Quick sanity check on your PC (optional)

You can test the TFLite model on your PC before moving to ESP32.

1. Install the runtime:
   - `python -m pip install tflite-runtime`
2. Use a small Python test script to load the TFLite model and run inference.

## 6) ESP32 inference overview

Important: The model expects a 2048-bit Morgan fingerprint as input. Computing that fingerprint requires RDKit, which is too heavy for ESP32. You have two options:

Option A (simplest):

- Compute the fingerprint on a PC and send the 2048 values to the ESP32 over serial/Wi-Fi.

Option B (fully on-device):

- Change the input feature to something you can compute on the ESP32.
- Retrain the model with that new feature.

If you want a minimal ESP32 inference example, see the TFLite Micro skeleton below.

## 7) ESP32 setup (beginner steps)

These steps use the Arduino IDE because it is the simplest option for beginners.

### 7.1 Install the Arduino IDE

1. Download Arduino IDE 2.x from <https://www.arduino.cc/en/software>.
2. Install it and launch the IDE.

### 7.2 Add ESP32 support

1. In Arduino IDE, open: File -> Preferences.
2. In "Additional Boards Manager URLs", paste:
   - <https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json>
3. Click OK.
4. Go to Tools -> Board -> Boards Manager.
5. Search for "esp32" and install "esp32 by Espressif Systems".

### 7.3 Connect your ESP32

1. Plug the ESP32 into your PC using a USB cable (data-capable, not charge-only).
2. Wait for Windows to install the USB driver.
   - If it fails, install the CP210x or CH340 driver (depends on your board).
3. In Arduino IDE, select your board:
   - Tools -> Board -> ESP32 Arduino -> your ESP32 board (example: "ESP32 Dev Module").
4. Select the COM port:
   - Tools -> Port -> COMx.

### 7.4 Add TFLite Micro

There are two common paths:

Option A (Arduino library):

1. Open Sketch -> Include Library -> Manage Libraries.
2. Search for "TensorFlowLite" and install the library by Arduino.

Option B (ESP-IDF component):

- Use the ESP-IDF build system and add the TFLite Micro component.
- This is more advanced; start with Option A if you are new.

### 7.5 Prepare the model for firmware

1. Generate a C array from the TFLite file:
   - `xxd -i .\artifacts\logp_model.tflite > model_data.h`
2. Move `model_data.h` next to your Arduino sketch or project source.
3. The generated array is named `logp_model_tflite`; rename it to `g_model_data`
   (or update the skeleton to use the generated name).

### 7.6 Minimal TFLite Micro skeleton

See `tflite_micro_skeleton.cpp` in this folder. It shows:

- How to load the model
- How to allocate the tensor arena
- How to call `Invoke()`

You still need to provide the 2048-element input vector (Morgan fingerprint).
On ESP32, compute that on a host PC and send the input over serial/Wi-Fi.

## 8) Files in this folder

- `train_logp_tflite.py`: Training script that exports a TFLite model.
- `tflite_test.cpp`: Placeholder C++ test file.
- `tflite_micro_skeleton.cpp`: Minimal ESP32 TFLite Micro inference skeleton.
- `Dataset/250k_rndm_zinc_drugs_clean_3.csv`: Training data.
