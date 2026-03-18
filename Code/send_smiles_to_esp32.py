"""Send a SMILES-derived Morgan fingerprint to the ESP32 and print the result.

Example:
    python Code/send_smiles_to_esp32.py "CCO"

Dependencies:
    rdkit
    pyserial
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial import SerialException

from mol_preprocessing import canonicalize_smiles, smiles_to_morgan_fingerprint


FP_BITS = 2048
FP_RADIUS = 2
DEFAULT_PORT = "COM4"
DEFAULT_BAUDRATE = 115200


def smiles_to_bitstring(smiles: str) -> str:
    fingerprint = smiles_to_morgan_fingerprint(smiles, radius=FP_RADIUS, n_bits=FP_BITS)
    if fingerprint is None:
        raise ValueError(f"Invalid or uncleanable SMILES: {smiles}")
    return "".join("1" if int(bit) else "0" for bit in fingerprint.tolist())


def expect_pong(connection: serial.Serial, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    connection.write(b"PING\n")
    connection.flush()

    while time.monotonic() < deadline:
        line = connection.readline().decode("ascii", errors="replace").strip()
        if not line:
            continue
        if line == "PONG":
            return
        print(f"ESP32> {line}")

    raise TimeoutError("Timed out waiting for PONG from the ESP32")


def request_prediction(
    connection: serial.Serial,
    bitstring: str,
    timeout_seconds: float,
) -> tuple[float, int]:
    command = f"FP {bitstring}\n".encode("ascii")
    connection.write(command)
    connection.flush()

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        line = connection.readline().decode("ascii", errors="replace").strip()
        if not line:
            continue

        if line.startswith("RESULT "):
            _, prediction_text, inference_us_text = line.split(maxsplit=2)
            return float(prediction_text), int(inference_us_text)

        if line.startswith("ERROR "):
            raise RuntimeError(f"ESP32 returned: {line}")

        print(f"ESP32> {line}")

    raise TimeoutError("Timed out waiting for a RESULT line from the ESP32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="?", help="SMILES string to predict")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port of the ESP32")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baudrate")
    parser.add_argument(
        "--boot-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after opening the serial port",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for ESP32 replies",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    smiles = args.smiles or input("SMILES> ").strip()
    if not smiles:
        print("No SMILES provided", file=sys.stderr)
        return 1

    try:
        bitstring = smiles_to_bitstring(smiles)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Connecting to {args.port} at {args.baudrate} baud...")
    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.2, write_timeout=2.0) as connection:
            time.sleep(args.boot_wait)
            connection.reset_input_buffer()
            connection.reset_output_buffer()

            expect_pong(connection, timeout_seconds=args.timeout)
            prediction, inference_us = request_prediction(
                connection,
                bitstring,
                timeout_seconds=args.timeout,
            )
    except SerialException as exc:
        print(f"Could not open {args.port}: {exc}", file=sys.stderr)
        print("Close anything else using the serial port and try again.", file=sys.stderr)
        print("Common port hogs: ESP-IDF monitor, VS Code serial monitor, Arduino Serial Monitor, PuTTY, and another Python process.", file=sys.stderr)
        print(f"If you just flashed the board, close the monitor first, then rerun: python Code/send_smiles_to_esp32.py --port {args.port} \"CCO\"", file=sys.stderr)
        return 1

    canonical_smiles = canonicalize_smiles(smiles)
    print(f"SMILES: {smiles}")
    print(f"Canonical cleaned SMILES: {canonical_smiles}")
    print(f"Predicted logP: {prediction:.6f}")
    print(f"ESP32 inference time: {inference_us} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())