# Conformer Ensemble Generation — Tier 1 and the xtb-Rescored Middle Ground

Three scripts, tested end-to-end:

- `conformer_ensemble_tier1.py` — RDKit ETKDGv3 embedding + MMFF94s
  optimization + RMSD deduplication + energy-window filtering + Boltzmann
  weighting. Usable standalone as the fast/practical tier.
- `conformer_ensemble_xtb_rescore.py` — takes a Tier 1 ensemble and rescores
  every surviving conformer with a GFN2-xTB single-point calculation using
  implicit water solvation (ALPB), then recomputes Boltzmann weights from
  those energies. This is the recommended middle ground.
- `run_batch.py` — parallel batch driver over a full molecule list (e.g. all
  of FreeSolv), with per-molecule error isolation and resume-after-crash.

## Setup

```bash
pip install rdkit --break-system-packages    # or just `pip install rdkit` in a venv

# xtb (only needed for the rescoring tier):
apt-get install -y xtb                        # Ubuntu/Debian — this is how it was tested here
# or, if you use conda/mamba:
# conda install -c conda-forge xtb
```

Both installs were verified in this environment: RDKit `2026.3.4`, xtb `6.6.1`.

## Quick single-molecule check

```bash
python3 conformer_ensemble_xtb_rescore.py "CCCCCC"    # hexane
```

This builds the MMFF94s ensemble, rescores it with GFN2-xTB/ALPB(water), and
prints each conformer's MMFF vs. xtb relative energy and final Boltzmann
weight side by side — a good way to eyeball whether the two energy functions
substantially reorder conformers for molecules you care about.

## Running over a full dataset (e.g. FreeSolv)

1. Prepare a CSV with at least `id,smiles` columns (FreeSolv's own
   `database.txt`/`.csv` gives you `iupac`/`smiles` directly — just extract
   those two columns; e.g. use the `mobley_...` compound ID as `id`).

2. Run the fast tier first, always — it's your correctness checkpoint and
   costs almost nothing:
   ```bash
   python3 run_batch.py freesolv.csv --tier tier1 --out ensembles_tier1.jsonl --workers 8
   ```

3. If you want the xtb-rescored middle ground, run it as a second pass
   (it re-embeds + re-optimizes with MMFF first, then rescores — this is
   intentional, so the two output files are independently reproducible):
   ```bash
   python3 run_batch.py freesolv.csv --tier xtb --out ensembles_xtb.jsonl --workers 8
   ```

4. If a run gets interrupted (crash, timeout, killed job), just re-run the
   same command — `run_batch.py` reads the existing output file, skips
   already-completed `id`s, and only processes what's left.

## Expected compute cost (measured in this environment)

- Tier 1 (RDKit/MMFF94s) alone: well under 1 second per molecule for
  typical FreeSolv-sized molecules (measured: ~0.6s for a 3-rotatable-bond
  molecule with 50 initial embeddings).
- xtb single-point rescoring: ~0.03s per conformer (GFN2-xTB, small
  molecule, ALPB water). For a molecule with ~10 surviving conformers,
  that's well under half a second of extra cost.
- **In practice, the entire FreeSolv set (~642 molecules) should complete
  in well under an hour on a single modern machine for either tier** — this
  is far cheaper than the original roadmap's conservative "hours to days"
  estimate for CREST; the xtb-*rescoring* step (single points only, no
  metadynamics search) is much cheaper than a full CREST conformer search
  would be. If you do want the full CREST treatment as a gold-standard
  robustness check on a subset of molecules, budget for that separately —
  it is a different (much more expensive) tool than xtb single points.

## Step 4: Coulomb Matrix descriptors and the (mu, sigma) input distribution

`compute_coulomb_descriptors.py` takes the JSONL output of `run_batch.py`
and produces exactly the per-molecule \((\mu_i, \sigma_i)\) pair that gets
fed into the GPyTorch uncertain-inputs training loop:

```bash
python3 compute_coulomb_descriptors.py ensembles_tier1.jsonl \
    --out descriptors_tier1.npz \
    --permutation eigenspectrum \
    --standardize
```

- **Default representation is `eigenspectrum`** (sorted eigenvalues of the
  Coulomb matrix), not the full `sorted_l2` matrix: it's permutation-,
  rotation-, and translation-invariant by construction, and gives a compact
  `n_atoms_max`-length vector instead of an `n_atoms_max^2` one. `sorted_l2`
  is available if you want the fuller representation instead.
- `n_atoms_max` is a **dataset-level** quantity (the largest molecule's atom
  count, across the whole dataset, including explicit Hs) — every
  molecule's descriptor vector must have the same length for a GP kernel to
  operate on them jointly, so this is computed automatically from all
  records in the input file before any per-molecule work starts.
- For each molecule, the script computes the Coulomb Matrix descriptor for
  every surviving conformer, then the **Boltzmann-weighted mean and
  standard deviation** across the ensemble (using the `weight` field each
  conformer already carries from ensemble generation) — this is \(\mu_i\)
  and \(\sigma_i\) directly.
- **Verified behavior**: molecules whose ensemble collapsed to a single
  conformer (rigid molecules — very common in FreeSolv) get \(\sigma_i = 0\)
  exactly, as they should; a flexible molecule like hexane gets a real,
  nonzero \(\sigma_i\). This was checked directly rather than assumed.
- **`--standardize` is recommended** by default: raw Coulomb matrix
  eigenvalues span a wide range (they scale with nuclear charge, roughly as
  \(Z^{2.4}\) on the diagonal terms feeding into them), and an
  RBF/Matérn-kernel GP is sensitive to input scale. `--standardize`
  z-scores \(\mu\) across the dataset (per dimension) and rescales
  \(\sigma\) by the *same* per-dimension factor — i.e. it's a location-and-
  scale transform on \(\mu\) but a scale-only transform on \(\sigma\) (a
  spread shouldn't get re-centered). This preserves each molecule's
  relative uncertainty magnitude while putting all dimensions on a
  comparable scale for the GP.
- Output is a single `.npz` with `ids`, `mu` (n_molecules × descriptor_dim),
  `sigma` (same shape), plus the `n_atoms_max`/`permutation`/`standardized`
  metadata used to produce it — load with `np.load(...)`, this is the
  direct input to the stochastic-block half of the GPyTorch training loop
  from the roadmap (concatenate with your fixed 2D block, then
  `torch.distributions.Normal(mu, sigma).rsample()` each step).

## Things worth checking before trusting the full run

- **Charge handling**: `run_batch.py` computes formal charge automatically
  from each SMILES via RDKit and passes it to xtb — don't hardcode
  `charge=0` if your dataset includes any charged species (check FreeSolv
  for this; most entries are neutral but verify rather than assume).
- **Failed embeddings**: `run_batch.py` isolates failures per-molecule and
  logs them to stderr rather than crashing the whole batch — check the
  stderr log afterwards for any `FAILED id=...` lines and inspect those
  molecules individually (unusual valences, macrocycles, and some charged/
  radical species are the usual culprits).
- **MMFF parameter availability**: `conformer_ensemble_tier1.py` falls back
  to UFF automatically if MMFF94s parameters aren't available for a given
  molecule (rare, but happens for some element/charge combinations) —
  worth flagging which molecules triggered this fallback if you want the
  energetics tier to be fully consistent across the dataset.
