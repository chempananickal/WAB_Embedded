"""Shared RDKit molecule cleanup utilities for training and inference."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def cleanup_molecule(molecule: Chem.Mol) -> Chem.Mol:
    cleaned_molecule = Chem.RemoveHs(Chem.Mol(molecule))

    canonical_smiles = Chem.MolToSmiles(
        cleaned_molecule,
        canonical=True,
        isomericSmiles=True,
    )
    canonical_molecule = Chem.MolFromSmiles(canonical_smiles)
    if canonical_molecule is None:
        raise ValueError(f"Could not parse canonical SMILES after cleanup: {canonical_smiles}")

    canonical_molecule = Chem.RemoveHs(canonical_molecule)
    return canonical_molecule


def smiles_to_clean_molecule(smiles: str) -> Chem.Mol | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    try:
        return cleanup_molecule(molecule)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str:
    molecule = smiles_to_clean_molecule(smiles)
    if molecule is None:
        raise ValueError(f"Invalid or uncleanable SMILES: {smiles}")

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def smiles_to_morgan_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    molecule = smiles_to_clean_molecule(smiles)
    if molecule is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)