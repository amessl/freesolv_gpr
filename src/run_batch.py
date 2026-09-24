"""
Batch driver: run Tier 1 (RDKit/MMFF94s) or the xtb-rescored middle-ground
pipeline over a list of molecules, in parallel across molecules, with
per-molecule error isolation and incremental checkpointing (so a crash or
timeout on one molecule doesn't lose earlier results, and you can resume).

Input format: a CSV with at least columns `id,smiles`. If FreeSolv (or any
other dataset) includes charged species, formal charge is computed
automatically from the SMILES via RDKit -- do not assume charge=0 globally.

Usage:
    python run_batch.py molecules.csv --tier tier1  --out ensembles_tier1.jsonl
    python run_batch.py molecules.csv --tier xtb    --out ensembles_xtb.jsonl --workers 8
"""

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem

from conformer_ensemble_tier1 import build_ensemble

from datasets import load_dataset

def process_one(mol_id: str, smiles: str, tier: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)

    result = build_ensemble(smiles)

    if tier == "xtb":
        # imported lazily inside the worker process so that ProcessPoolExecutor
        # workers each get their own xtb subprocess handling correctly
        from src.conformer_ensemble_xtb_rescore import rescore_with_xtb
        result = rescore_with_xtb(result, charge=charge)

    result["id"] = mol_id
    result["formal_charge"] = charge
    return result


def already_done(out_path: str) -> set:
    """Read an existing output file (if resuming) to skip completed molecules."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["id"])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("input_csv", help="CSV with columns: id,smiles")
    ap.add_argument("--dataset", choices=["freesolv", "esol", "lipo"])

    ap.add_argument("--tier", choices=["tier1", "xtb"], default="tier1")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    dataset_dict = {"freesolv": "scikit-fingerprints/MoleculeNet_FreeSolv",
                    "esol": "scikit-fingerprints/MoleculeNet_ESOL",
                    "lipo": "scikit-fingerprints/MoleculeNet_Lipophilicity"}

    dataset = load_dataset(dataset_dict[args.dataset], split="train")
    dataset.to_csv(f"../data/{args.dataset}.csv")

    with open(f"../data/{args.dataset}.csv") as f:
        reader = csv.DictReader(f)
        molecules = [(i, row["SMILES"]) for i, row in enumerate(reader)]

    done = already_done(args.out)
    todo = [(mid, smi) for mid, smi in molecules if mid not in done]
    print(f"{len(molecules)} total, {len(done)} already done, {len(todo)} to process",
          file=sys.stderr)

    out_f = open(args.out, "a")
    n_ok, n_fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, mid, smi, args.tier): (mid, smi)
            for mid, smi in todo
        }
        for fut in as_completed(futures):
            mid, smi = futures[fut]
            try:
                result = fut.result(timeout=600)
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                print(f"FAILED id={mid} smiles={smi}: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            if (n_ok + n_fail) % 25 == 0:
                print(f"  progress: {n_ok} ok, {n_fail} failed, "
                      f"{len(todo) - n_ok - n_fail} remaining", file=sys.stderr)

    out_f.close()
    print(f"Done. {n_ok} succeeded, {n_fail} failed. Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
