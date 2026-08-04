"""
Recommended middle-ground pipeline: take the RDKit/MMFF94s ensemble from
conformer_ensemble_tier1.py, and rescore each surviving conformer's energy
with a single-point GFN2-xTB calculation using implicit water solvation
(ALPB model), then recompute Boltzmann weights from the xtb energies.

Requires the `xtb` binary on PATH (e.g. `apt install xtb`, or
`conda install -c conda-forge xtb`).

Usage:
    from conformer_ensemble_tier1 import build_ensemble
    from conformer_ensemble_xtb_rescore import rescore_with_xtb

    result = build_ensemble("CCCCCC")
    rescored = rescore_with_xtb(result)
"""

import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
from rdkit import Chem

R_KCAL = 0.0019872041
T_DEFAULT = 298.15
HARTREE_TO_KCAL = 627.5094740631

XTB_BINARY = shutil.which("xtb")


def _molblock_to_xyz(molblock: str) -> str:
    """Convert an RDKit molblock to an XYZ block via a temporary mol object
    (keeps this module independent of writing files for the conversion)."""
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), ""]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<3s}{pos.x:>14.8f}{pos.y:>14.8f}{pos.z:>14.8f}")
    return "\n".join(lines) + "\n"


def _run_xtb_singlepoint(xyz_block: str, charge: int = 0, solvent: str = "water",
                          workdir: str = None) -> float:
    """Run a single-point GFN2-xTB calculation with ALPB implicit solvent.
    Returns the total energy in Hartree, parsed from xtb's stdout."""
    if XTB_BINARY is None:
        raise RuntimeError(
            "xtb binary not found on PATH. Install it, e.g. "
            "'apt install xtb' or 'conda install -c conda-forge xtb'."
        )

    own_tmp = workdir is None
    if own_tmp:
        workdir = tempfile.mkdtemp(prefix="xtb_sp_")

    xyz_path = os.path.join(workdir, "mol.xyz")
    with open(xyz_path, "w") as f:
        f.write(xyz_block)

    cmd = [
        XTB_BINARY, "mol.xyz",
        "--gfn", "2",
        "--alpb", solvent,
        "--chrg", str(charge),
        "--sp",  # single-point only, no geometry optimization
    ]
    proc = subprocess.run(
        cmd, cwd=workdir, capture_output=True, text=True, timeout=300
    )

    if own_tmp:
        stdout = proc.stdout
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        stdout = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(f"xtb failed:\n{proc.stdout}\n{proc.stderr}")

    # xtb prints a line like: "          | TOTAL ENERGY             -12.345678900 Eh   |"
    match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", stdout)
    if not match:
        raise RuntimeError(f"Could not parse xtb energy from output:\n{stdout}")
    return float(match.group(1))


def rescore_with_xtb(ensemble_result: dict, solvent: str = "water",
                      charge: int = 0, T: float = T_DEFAULT) -> dict:
    """Take the output of conformer_ensemble_tier1.build_ensemble and rescore
    every surviving conformer's energy with GFN2-xTB + ALPB(water), then
    recompute Boltzmann weights from the rescored energies.

    Note: charge must be supplied per-molecule if any FreeSolv entries are
    charged species (most are neutral, but check before assuming charge=0
    for the full dataset)."""
    xtb_energies_hartree = []
    for conf in ensemble_result["conformers"]:
        xyz = _molblock_to_xyz(conf["molblock"])
        e_hartree = _run_xtb_singlepoint(xyz, charge=charge, solvent=solvent)
        xtb_energies_hartree.append(e_hartree)

    xtb_energies_hartree = np.array(xtb_energies_hartree)
    e_min = xtb_energies_hartree.min()
    rel_energies_kcal = (xtb_energies_hartree - e_min) * HARTREE_TO_KCAL

    boltz = np.exp(-rel_energies_kcal / (R_KCAL * T))
    weights = boltz / boltz.sum()

    rescored_conformers = []
    for conf, e_rel, w in zip(ensemble_result["conformers"], rel_energies_kcal, weights):
        new_conf = dict(conf)
        new_conf["mmff_energy_kcal_mol"] = conf["energy_kcal_mol"]  # keep original for comparison
        new_conf["energy_kcal_mol"] = float(e_rel)  # now xtb-relative energy
        new_conf["weight"] = float(w)
        rescored_conformers.append(new_conf)

    result = dict(ensemble_result)
    result["conformers"] = rescored_conformers
    result["rescored_with"] = f"GFN2-xTB//ALPB({solvent})"
    return result


if __name__ == "__main__":
    import sys
    from conformer_ensemble_tier1 import build_ensemble

    smiles = sys.argv[1] if len(sys.argv) > 1 else "CCCCCC"
    print(f"Building MMFF94s ensemble for {smiles} ...", file=sys.stderr)
    base = build_ensemble(smiles, verbose=True)
    print(f"Rescoring {base['n_final']} conformers with GFN2-xTB/ALPB(water) ...", file=sys.stderr)
    rescored = rescore_with_xtb(base)

    for c in rescored["conformers"]:
        print(
            f"  MMFF rel. E = {c['mmff_energy_kcal_mol']:.3f} kcal/mol  ->  "
            f"xtb rel. E = {c['energy_kcal_mol']:.3f} kcal/mol   "
            f"weight(mmff-based order) = {c['weight']:.3f}"
        )
