"""Render a single molecule with RDKit and save it as a PNG image."""

from __future__ import annotations

import argparse
from pathlib import Path

from rdkit.Chem import Draw

from mol_preprocessing import canonicalize_smiles, smiles_to_clean_molecule


DEFAULT_SMILES = "O=C([O-])C(O)(C(=O)[O-])C(=O)[O-]" # Citrate, the worst case molecule
DEFAULT_OUTPUT = Path(__file__).parent / "Results" / "molecules" / "worst_case.png"
IMAGE_SIZE = (1400, 1000)


def render_molecule(smiles: str, output_path: Path) -> Path:
    canonical_smiles = canonicalize_smiles(smiles)
    molecule = smiles_to_clean_molecule(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Invalid or uncleanable SMILES: {smiles}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolToImage(molecule, size=IMAGE_SIZE, kekulize=True)
    image.save(output_path)
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="?", default=DEFAULT_SMILES, help="SMILES string to render")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path of the output PNG file")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output_path = render_molecule(args.smiles, args.output)
    print(f"Saved molecule image to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())