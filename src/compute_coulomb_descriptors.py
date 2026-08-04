"""
Compute Coulomb Matrix descriptors (via DScribe) for every conformer in an
ensemble file produced by run_batch.py, then aggregate them into a
per-molecule Boltzmann-weighted mean/std -- this (mu, sigma) pair is exactly
the per-molecule input distribution to feed into the GPyTorch
uncertain-inputs formalism (see conformer_ensemble_tier1.py /
conformer_ensemble_xtb_rescore.py for how the ensembles + weights were made).

Why "eigenspectrum" and not the full sorted Coulomb matrix by default:
- The full matrix scales as n_atoms_max^2 and needs a fixed atom ordering
  convention (sorted_l2) to be permutation-invariant, which is usable but
  large and partly redundant.
- The eigenspectrum (sorted eigenvalues of the Coulomb matrix) is fully
  permutation-invariant *and* invariant to rotation/translation by
  construction, and gives a compact n_atoms_max-length vector -- a natural
  fit for "one stochastic descriptor block per molecule." `sorted_l2` is
  still available via --permutation if you want the fuller representation.

Usage:
    python3 compute_coulomb_descriptors.py ensembles_tier1.jsonl \
        --out descriptors_tier1.npz \
        --permutation eigenspectrum \
        --standardize
"""

import json
import sys

import numpy as np
from ase import Atoms
from dscribe.descriptors import CoulombMatrix
from rdkit import Chem
from hydra import initialize, compose


def molblock_to_ase_atoms(molblock: str) -> Atoms:
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=symbols, positions=positions)


def determine_n_atoms_max(records: list) -> int:
    """Coulomb matrices need one fixed size across the *entire* dataset
    (every molecule's descriptor vector must have identical length for the
    GP kernel to operate on them jointly), so this is a dataset-level
    quantity, not a per-molecule one."""
    n_max = 0
    for rec in records:
        for conf in rec["conformers"]:
            mol = Chem.MolFromMolBlock(conf["molblock"], removeHs=False)
            n_max = max(n_max, mol.GetNumAtoms())
    return n_max


def compute_descriptors_for_molecule(record: dict, cm: CoulombMatrix) -> np.ndarray:
    """Returns an (n_conformers, descriptor_dim) array for one molecule."""
    atoms_list = [molblock_to_ase_atoms(c["molblock"]) for c in record["conformers"]]
    descriptors = np.asarray(cm.create(atoms_list, n_jobs=1))
    if descriptors.ndim == 1:
        # dscribe squeezes to a flat 1D array when given a single-Atoms
        # list (i.e. a molecule whose ensemble collapsed to one conformer)
        # -- restore the (n_conformers, descriptor_dim) shape explicitly.
        descriptors = descriptors.reshape(1, -1)
    return descriptors


def weighted_mean_std(descriptors: np.ndarray, weights: np.ndarray):
    """Boltzmann-weighted mean and standard deviation per descriptor
    dimension. `weights` must already sum to 1 (they do, by construction,
    coming out of the ensemble-generation scripts)."""
    mu = np.average(descriptors, axis=0, weights=weights)
    var = np.average((descriptors - mu) ** 2, axis=0, weights=weights)
    sigma = np.sqrt(var)
    return mu, sigma


def main(cfg):

    with open(cfg.reps.ensemble_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} molecules", file=sys.stderr)

    n_atoms_max = determine_n_atoms_max(records)
    print(f"n_atoms_max across dataset = {n_atoms_max}", file=sys.stderr)

    cm = CoulombMatrix(n_atoms_max=n_atoms_max, permutation=cfg.reps.coulomb.permutation)

    ids, mus, sigmas = [], [], []
    for i, record in enumerate(records):
        weights = np.array([c["weight"] for c in record["conformers"]])
        weights = weights / weights.sum()  # re-normalize defensively

        descriptors = compute_descriptors_for_molecule(record, cm)
        if cfg.reps.coulomb.single_conformer:
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

    if cfg.reps.coulomb.standardize:
        # dataset-level per-dimension standardization: subtract the global
        # mean of mu, divide by the global std of mu. Sigma is *scaled* by
        # the same per-dimension factor (not re-centered -- it's a spread,
        # not a location) so it still represents "how much this molecule's
        # descriptor varies, in standardized units."
        global_mean = mus.mean(axis=0)
        global_std = mus.std(axis=0)
        global_std[global_std == 0] = 1.0  # avoid divide-by-zero on constant dims
        mus = (mus - global_mean) / global_std
        sigmas = sigmas / global_std
        print("Applied dataset-level standardization to mu; scaled sigma "
              "by the same per-dimension factor.", file=sys.stderr)

    np.savez(
        cfg.reps.coulomb.out,
        ids=np.array(ids),
        mu=mus,
        sigma=sigmas,
        n_atoms_max=n_atoms_max,
        permutation=cfg.reps.coulomb.permutation,
        standardized=cfg.reps.coulomb.standardize,
        single_conformer=cfg.reps.coulomb.single_conformer,
    )
    print(f"Saved {mus.shape[0]} molecules x {mus.shape[1]}-dim descriptors "
          f"to {cfg.reps.coulomb.out}", file=sys.stderr)


if __name__ == "__main__":

    with initialize(config_path="../conf", version_base="1.1"):
        cfg=compose(config_name="config")

    main(cfg)
