"""Generate a QR code from a standardized 2048-bit ECFP4 fingerprint.

Example:
    python Code/generate_fingerprint_qr.py "CCO" ethanol

Dependencies:
    rdkit
    qrcode
    pillow
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import qrcode

from mol_preprocessing import canonicalize_smiles, smiles_to_morgan_fingerprint


FP_BITS = 2048
FP_RADIUS = 2
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "Results" / "qr_codes"
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def sanitize_filename(name: str) -> str:
    cleaned_name = INVALID_FILENAME_CHARS.sub("_", name).strip(" ._")
    if not cleaned_name:
        raise ValueError("Name must contain at least one filename-safe character")
    return cleaned_name


def fingerprint_to_bitstring(smiles: str) -> tuple[str, str]:
    canonical_smiles = canonicalize_smiles(smiles)
    fingerprint = smiles_to_morgan_fingerprint(canonical_smiles, radius=FP_RADIUS, n_bits=FP_BITS)
    if fingerprint is None:
        raise ValueError(f"Invalid or uncleanable SMILES: {smiles}")

    bitstring = "".join("1" if int(bit) else "0" for bit in fingerprint.tolist())
    return canonical_smiles, bitstring


def bitstring_to_qr_payload(bitstring: str) -> str:
    packed_bits = np.packbits(np.fromiter((int(bit) for bit in bitstring), dtype=np.uint8))
    return base64.urlsafe_b64encode(packed_bits.tobytes()).decode("ascii").rstrip("=")


def qr_payload_to_bitstring(payload: str) -> str:
    padding = "=" * (-len(payload) % 4)
    packed_bits = np.frombuffer(base64.urlsafe_b64decode((payload + padding).encode("ascii")), dtype=np.uint8)
    unpacked_bits = np.unpackbits(packed_bits)
    if unpacked_bits.size < FP_BITS:
        raise ValueError(f"Decoded fingerprint is too short: {unpacked_bits.size} bits")
    return "".join("1" if int(bit) else "0" for bit in unpacked_bits[:FP_BITS].tolist())


def save_qr_code(bitstring: str, output_path: Path) -> None:
    payload = bitstring_to_qr_payload(bitstring)

    qr_code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, border=8, box_size=12)
    qr_code.add_data(payload)
    qr_code.make(fit=True)

    image = qr_code.make_image(fill_color="black", back_color="white")
    pil_image = image.get_image() if hasattr(image, "get_image") else image
    pil_image.save(output_path)


def decode_qr_payload(image: np.ndarray) -> str:
    detector = cv2.QRCodeDetector()
    decode_inputs = [
        image,
        cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
        cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST),
    ]

    for decode_input in decode_inputs:
        decoded_payload, _, _ = detector.detectAndDecode(decode_input)
        if decoded_payload:
            return decoded_payload

    raise ValueError("Could not decode QR payload from generated image")


def verify_qr_code(output_path: Path, expected_bitstring: str) -> None:
    image = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read generated QR image: {output_path}")

    decoded_payload = decode_qr_payload(image)
    decoded_bitstring = qr_payload_to_bitstring(decoded_payload)
    if decoded_bitstring != expected_bitstring:
        raise ValueError("Decoded QR fingerprint does not match the original bit vector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="?", help="SMILES string to encode")
    parser.add_argument("name", nargs="?", help="Name used for the saved QR image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the QR code image will be saved",
    )
    parser.add_argument(
        "--format",
        choices=["png"], # TODO: Add PDF and SVG support
        default="png",
        help="Output image format",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    smiles = args.smiles or input("SMILES> ").strip()
    if not smiles:
        print("No SMILES provided", file=sys.stderr)
        return 1

    name = args.name or input("Name> ").strip()
    if not name:
        print("No name provided", file=sys.stderr)
        return 1

    try:
        canonical_smiles, bitstring = fingerprint_to_bitstring(smiles)
        safe_name = sanitize_filename(name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{safe_name}.{args.format}"
    save_qr_code(bitstring, output_path)
    verify_qr_code(output_path, bitstring)

    print(f"Input SMILES: {smiles}")
    print(f"Canonical cleaned SMILES: {canonical_smiles}")
    print(f"Fingerprint length: {len(bitstring)} bits")
    print(f"Set bits: {bitstring.count('1')}")
    print("QR verification: passed")
    print(f"Saved QR code to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())