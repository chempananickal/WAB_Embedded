# ESP32 logP Inference

This folder is the actual ESP-IDF project for the model.

The workflow is simple:

1. The external device turns a SMILES string into a 2048-bit Morgan fingerprint.
2. The external device sends those 2048 bits over COM4 (or whatever COM port the ESP32 is connected to).
3. The ESP32 runs the LiteRT Micro model.
4. The ESP32 sends the predicted logP value back.
5. The Python script prints the answer in the terminal.

## What is in here

- `main/main.cc`: ESP32 firmware that loads the quantized model and listens on serial.
- `idf_component.yml`: pulls in Espressif's `esp-tflite-micro` component.
- `../../artifacts/logp_model_tflite.cc`: the model as a C array.

## The serial protocol

The ESP32 understands three commands:

- `PING`
- `HELP`
- `FP <2048 characters made of only 0 and 1>`

Example:

```text
FP 01001010...2048 total bits...
```

The board answers with one of these:

- `PONG`
- `INFO ...`
- `RESULT <predicted_logP> <inference_time_us>`
- `ERROR ...`

## Step by step

1. Open the folder `Code/esp32_logp` in VS Code, or make it the active ESP-IDF project.
2. Let ESP-IDF install the dependency from `idf_component.yml` the first time you build.
3. Build the project.
4. Flash it to the ESP32 on COM4.
5. Open serial monitor once to confirm you see a line starting with `READY`.
6. On the PC, run `send_smiles_to_esp32.py`.
7. Type or pass a SMILES string.
8. Read the printed logP prediction.

## Important detail

the model input is quantized int8.

- Fingerprint bit `0` becomes `-128` on the ESP32.
- Fingerprint bit `1` becomes `127` on the ESP32.

That matches the model input quantization:

- input scale = `0.003921568859`
- input zero point = `-128`

The output is also int8 and is converted back to float on the ESP32 before it is printed.

## If something goes wrong

- `ERROR allocate_tensors_failed`: increase `kTensorArenaSize` in `main/main.cc`.
- `ERROR unknown_command`: the Python side did not send the expected line.
- `ERROR fingerprint_length`: the host did not send exactly 2048 bits.
- No response after opening serial: the board may have rebooted; wait 2 seconds and try again.