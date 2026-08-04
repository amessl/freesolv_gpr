"""
Compute SOAP descriptors (via DScribe) for every conformer in an
ensemble file produced by run_batch.py, then aggregate them into a
per-molecule Boltzmann-weighted mean/std -- this (mu, sigma) pair is exactly
the per-molecule input distribution to feed into the GPyTorch
uncertain-inputs formalism (see conformer_ensemble_tier1.py /
conformer_ensemble_xtb_rescore.py for how the ensembles + weights were made).

This mirrors compute_coulomb_descriptors.py but uses DScribe's SOAP.

Usage:
    python3 compute_soap_descriptors.py ensembles_tier1.jsonl \
        --out descriptors_soap_tier1.npz \
        --rcut 5.0 --nmax 8 --lmax 6 --sigma 0.5 \
        --standardize
"""

import argparse
import json
import sys
from typing import List, Set
from hydra import initialize, compose

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from rdkit import Chem


def molblock_to_ase_atoms(molblock: str) -> Atoms:
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=symbols, positions=positions)


def determine_species(records: List[dict]) -> List[str]:
    """Collect the unique atomic species appearing across the dataset.

    SOAP requires a fixed list of species for the whole dataset.
    """
    species: Set[str] = set()
    for rec in records:
        for conf in rec["conformers"]:
            mol = Chem.MolFromMolBlock(conf["molblock"], removeHs=False)
            for atom in mol.GetAtoms():
                species.add(atom.GetSymbol())
    # DScribe expects a list; sort for reproducibility
    return sorted(species)


def compute_descriptors_for_molecule(record: dict, soap: SOAP) -> np.ndarray:
    """Returns an (n_conformers, descriptor_dim) array for one molecule."""
    atoms_list = [molblock_to_ase_atoms(c["molblock"]) for c in record["conformers"]]
    descriptors = np.asarray(soap.create(atoms_list, n_jobs=1))
    if descriptors.ndim == 1:
        # DScribe squeezes to a flat 1D array when given a single-Atoms list
        descriptors = descriptors.reshape(1, -1)
    return descriptors


essential_float = float  # alias for type hints in argparse defaults


def weighted_mean_std(descriptors: np.ndarray, weights: np.ndarray):
    """Boltzmann-weighted mean and standard deviation per descriptor dim.
    `weights` must sum to 1.
    """
    mu = np.average(descriptors, axis=0, weights=weights)
    var = np.average((descriptors - mu) ** 2, axis=0, weights=weights)
    sigma = np.sqrt(var)
    return mu, sigma


def main(cfg):
    with open(cfg.reps.ensemble_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} molecules", file=sys.stderr)

    species = determine_species(records)
    print(f"Species across dataset = {species}", file=sys.stderr)

    props = {"H": 2.20,
             "C": 2.55,
             "N": 3.04,
             "O": 3.44,
             "S": 2.58,
             "F": 3.98,
             "Cl": 3.16,
             "Br": 2.96,
             "I": 2.66,
             "P": 2.19}

    if cfg.reps.soap.compress:
        compression = {"mode": "mu2",
                       "species_weighting": props}
    else:
        compression = {"mode": "off"}

    # Use vector SOAP by averaging over atoms to obtain a fixed-length per-structure descriptor
    soap = SOAP(species=species, r_cut=cfg.reps.soap.rcut, n_max=cfg.reps.soap.nmax,
                l_max=cfg.reps.soap.lmax, sigma=cfg.reps.soap.sigma, periodic=False, average='outer',
                compression=compression)

    ids, mus, sigmas = [], [], []
    for i, record in enumerate(records):
        weights = np.array([c["weight"] for c in record["conformers"]])
        weights = weights / weights.sum()  # re-normalize defensively

        descriptors = compute_descriptors_for_molecule(record, soap)
        if cfg.reps.soap.single_conformer:
            # Use the single conformer with highest Boltzmann weight
            idx_best = int(np.argmax(weights))
            mu = descriptors[idx_best]
            sigma = np.zeros_like(mu)
        else:
            mu, sigma = weighted_mean_std(descriptors, weights)

        ids.append(record["id"])
        mus.append(mu)
        sigmas.append(sigma)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} molecules processed", file=sys.stderr)

    mus = np.vstack(mus)       # (n_molecules, descriptor_dim)
    sigmas = np.vstack(sigmas)  # (n_molecules, descriptor_dim)

    if cfg.reps.soap.standardize:
        # dataset-level per-dimension standardization: subtract the global
        # mean of mu, divide by the global std of mu. Sigma is scaled by
        # the same per-dimension factor (not re-centered).
        global_mean = mus.mean(axis=0)
        global_std = mus.std(axis=0)
        global_std[global_std == 0] = 1.0  # avoid divide-by-zero on constant dims
        mus = (mus - global_mean) / global_std
        sigmas = sigmas / global_std
        print("Applied dataset-level standardization to mu; scaled sigma by the same per-dimension factor.", file=sys.stderr)

    np.savez(
        cfg.reps.soap.out,
        ids=np.array(ids),
        mu=mus,
        sigma=sigmas,
        species=np.array(species),
        soap_params=dict(rcut=cfg.reps.soap.rcut, nmax=cfg.reps.soap.nmax,
                         lmax=cfg.reps.soap.lmax, sigma=cfg.reps.soap.sigma),
        standardized=cfg.reps.soap.standardize,
        single_conformer=cfg.reps.soap.single_conformer,
    )
    print(f"Saved {mus.shape[0]} molecules x {mus.shape[1]}-dim SOAP descriptors to {cfg.reps.soap.out}", file=sys.stderr)


if __name__ == "__main__":

    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config")

    main(cfg)
