"""
Tier 1 (fast/practical) conformer ensemble generation:
RDKit ETKDGv3 embedding -> MMFF94s optimization -> RMSD pruning ->
energy-window filtering -> Boltzmann weighting.

Usage (as a library):
    from conformer_ensemble_tier1 import build_ensemble
    result = build_ensemble("CCO")  # ethanol
    for conf in result["conformers"]:
        print(conf["energy_kcal_mol"], conf["weight"])

Usage (as a script, one SMILES per line in input file):
    python conformer_ensemble_tier1.py molecules.smi > ensembles.json
"""

import sys
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdMolAlign

R_KCAL = 0.0019872041          # kcal / (mol K)
T_DEFAULT = 298.15             # K, matches FreeSolv reference conditions
ENERGY_WINDOW_KCAL = 4.0        # discard conformers above min + this window
RMSD_DUPLICATE_THRESHOLD = 0.5  # Angstrom, heavy-atom RMSD after alignment


def n_confs_from_flexibility(mol: Chem.Mol) -> int:
    """Adaptive initial embedding count, scaled by rotatable-bond count.
    (Rough heuristic from RDKit conformer-generation benchmarking practice;
    treat as a tunable starting point, not a fixed law.)"""
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot <= 3:
        return 50
    elif n_rot <= 6:
        return 100
    elif n_rot <= 9:
        return 200
    else:
        return 300


def embed_and_optimize(mol: Chem.Mol, n_confs: int, seed: int = 0xF00D):
    """Embed n_confs conformers with ETKDGv3, optimize each with MMFF94s.
    Returns the mol (with conformers attached) and a list of energies (kcal/mol),
    aligned by conformer id order. Failed optimizations are marked None and
    filtered out by the caller."""
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1  # we do our own RMSD pruning after optimization
    params.useRandomCoords = True
    params.numThreads = 0  # use all available

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    energies = {}
    # MMFF94s optimization; MMFFOptimizeMoleculeConfs returns (not_converged, energy) per conf
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if mmff_props is None:
        # MMFF parameters unavailable for this molecule (rare, e.g. some
        # exotic elements) -- fall back to UFF
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        for cid, (not_converged, energy) in zip(conf_ids, results):
            energies[cid] = energy if not_converged == 0 else None
    else:
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, mmffVariant="MMFF94s", maxIters=2000
        )
        for cid, (not_converged, energy) in zip(conf_ids, results):
            energies[cid] = energy if not_converged == 0 else None

    return mol, energies


def prune_duplicates(mol: Chem.Mol, energies: dict, rmsd_threshold=RMSD_DUPLICATE_THRESHOLD):
    """Greedy RMSD-based deduplication: sort by energy, keep a conformer only
    if its heavy-atom RMSD to every already-kept, lower-energy conformer
    exceeds the threshold."""
    valid_ids = [cid for cid, e in energies.items() if e is not None]
    valid_ids.sort(key=lambda cid: energies[cid])

    heavy_atom_mol = Chem.RemoveHs(mol)  # RMSD on heavy atoms only
    kept = []
    for cid in valid_ids:
        is_dup = False
        for kept_cid in kept:
            try:
                rmsd = rdMolAlign.GetBestRMS(
                    heavy_atom_mol, heavy_atom_mol, prbId=cid, refId=kept_cid
                )
            except Exception:
                rmsd = rdMolAlign.AlignMol(mol, mol, prbCid=cid, refCid=kept_cid)
            if rmsd < rmsd_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(cid)
    return kept


def filter_by_energy_window(kept_ids, energies, window_kcal=ENERGY_WINDOW_KCAL):
    e_min = min(energies[cid] for cid in kept_ids)
    return [cid for cid in kept_ids if energies[cid] - e_min <= window_kcal], e_min


def boltzmann_weights(kept_ids, energies, e_min, T=T_DEFAULT):
    rel_energies = np.array([energies[cid] - e_min for cid in kept_ids])
    boltz = np.exp(-rel_energies / (R_KCAL * T))
    weights = boltz / boltz.sum()
    return weights


def build_ensemble(smiles: str, T: float = T_DEFAULT, verbose: bool = False) -> dict:
    """Full Tier 1 pipeline for one molecule. Returns a dict with per-conformer
    3D coordinates (as RDKit mol block), relative energies, and Boltzmann weights."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    n_confs = n_confs_from_flexibility(mol)
    mol, energies = embed_and_optimize(mol, n_confs)

    n_embedded = len(energies)
    n_converged = sum(1 for e in energies.values() if e is not None)
    if n_converged == 0:
        raise RuntimeError(f"No conformers converged for {smiles}")

    kept_ids = prune_duplicates(mol, energies)
    kept_ids, e_min = filter_by_energy_window(kept_ids, energies)
    weights = boltzmann_weights(kept_ids, energies, e_min, T=T)

    if verbose:
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveHs(mol))
        print(
            f"{smiles}: {n_rot} rotatable bonds, {n_embedded} embedded, "
            f"{n_converged} converged, {len(kept_ids)} kept after dedup+energy window",
            file=sys.stderr,
        )

    conformers = []
    for cid, w in zip(kept_ids, weights):
        conformers.append({
            "conf_id": int(cid),
            "energy_kcal_mol": float(energies[cid] - e_min),  # relative to min
            "weight": float(w),
            "molblock": Chem.MolToMolBlock(mol, confId=cid),
        })

    return {
        "smiles": smiles,
        "n_rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveHs(mol))),
        "n_embedded": n_embedded,
        "n_converged": n_converged,
        "n_final": len(kept_ids),
        "conformers": conformers,
    }


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else None
    if infile:
        with open(infile) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        smiles_list = [line.strip() for line in sys.stdin if line.strip()]

    results = []
    for smi in smiles_list:
        try:
            res = build_ensemble(smi, verbose=True)
            results.append(res)
        except Exception as exc:
            print(f"FAILED on {smi}: {exc}", file=sys.stderr)

    print(json.dumps(results))
