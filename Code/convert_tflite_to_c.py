"""Convert a .tflite file into C source/header (xxd -i style).

Usage: python Code/convert_tflite_to_c.py
It reads artifacts/logp_model.tflite and writes:
 - artifacts/logp_model_tflite.cc
 - artifacts/logp_model_tflite.h
"""
from pathlib import Path

IN_PATH = Path(__file__).parent / "artifacts" / "logp_model.tflite"
OUT_CC = Path(__file__).parent / "artifacts" / "logp_model_tflite.cc"
OUT_H = Path(__file__).parent / "artifacts" / "logp_model_tflite.h"

def to_c_array(name: str, data: bytes) -> str:
    """
    Convert bytes to a C array definition with the given variable name.
    Emulates the style of xxd -i output, with 12 hex bytes per line and a separate length variable.
    """
    lines = []
    lines.append(f"const unsigned char {name}[] = {{")
    hex_bytes = [f"0x{b:02x}" for b in data]
    # wrap columns at 12 entries
    for i in range(0, len(hex_bytes), 12):
        lines.append("    " + ", ".join(hex_bytes[i:i+12]) + ("," if i+12 < len(hex_bytes) else ""))
    lines.append("};")
    lines.append(f"const unsigned int {name}_len = {len(data)};")
    return "\n".join(lines)

def main():
    if not IN_PATH.exists():
        print(f"Input file not found: {IN_PATH}")
        return

    data = IN_PATH.read_bytes()

    varname = "logp_model_tflite"
    cc = []
    cc.append("#include \"logp_model_tflite.h\"")
    cc.append("")
    cc.append(to_c_array(varname, data))
    cc_text = "\n".join(cc) + "\n"

    h = []
    h.append("#pragma once")
    h.append("")
    h.append("#include <stdint.h>")
    h.append("")
    h.append(f"extern const unsigned char {varname}[];")
    h.append(f"extern const unsigned int {varname}_len;")
    h_text = "\n".join(h) + "\n"

    OUT_CC.write_text(cc_text)
    OUT_H.write_text(h_text)

    print(f"Wrote {OUT_CC}")
    print(f"Wrote {OUT_H}")

if __name__ == '__main__':
    main()
