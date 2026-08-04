# Conformer-Ensemble-Derived Input Uncertainty for Gaussian Process Regression
## Publication Concept & Research Roadmap (v2 — refined around GPyTorch's SVI-with-uncertain-inputs formalism)

---

## 0. What changed from v1, and why it matters

The previous draft built the story around *two* physically-grounded noise
terms (input noise + heteroscedastic label noise) and a McHutchon–Rasmussen
(NIGP)-style analytic noise-propagation scheme. You've now specified the
actual mechanism you want to use:

**[GPyTorch — "GP Regression with Uncertain Inputs"](https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/GP_Regression_with_Uncertain_Inputs.html)**

The key mechanics of that formalism, concretely:
- Each training input is represented as a distribution, not a point:
  \(x_i \sim \mathcal{N}(\mu_i, \sigma_i)\).
- The model is an `ApproximateGP` (a sparse/inducing-point variational GP —
  `VariationalStrategy` + `CholeskyVariationalDistribution`), trained with
  `VariationalELBO`.
- At **every optimization step**, a fresh sample `x_sample = Normal(mu, std).rsample()`
  is drawn (reparameterization trick → differentiable), the ELBO is computed
  on that sample, and stochastic-gradient optimization (Adam) converges to
  (a local optimum of) the *true* ELBO in expectation over the input
  distribution.
- Label noise is handled by a plain `GaussianLikelihood` — homoscedastic,
  learned as a single global hyperparameter. **No heteroscedastic
  likelihood is used anywhere in this formalism.**

This resolves the scope question directly: you are **not** building a
heteroscedastic-noise model. You are using the conformer ensemble purely to
supply \((\mu_i, \sigma_i)\) — the per-molecule input distribution — and
letting the existing SVI machinery propagate that uncertainty through
training via Monte Carlo resampling rather than analytic linearization. This
is simpler to implement than NIGP, has no gradient-of-the-mean-function
bootstrapping loop, and is already there in GPyTorch off-the-shelf.

Everything below is restructured around this.

---

## 1. Core Idea (restated)

A molecule at room temperature is a Boltzmann ensemble of conformers, so any
3D-derived descriptor is a random variable, not a fixed number. Rather than
inventing a noise model for this, we **measure it**: run a conformer
ensemble, compute the empirical (Boltzmann-weighted) mean and per-dimension
standard deviation of the conformer-sensitive descriptors, and feed those
directly as \((\mu_i, \sigma_i)\) into a variational GP trained exactly as
in the GPyTorch tutorial — i.e., resampling the input at every SVI step.
Label noise stays a standard homoscedastic Gaussian likelihood; the paper's
job is to show that propagating a *physically measured* input distribution
this way improves calibration (and study when/why), not to reinvent noise
modeling on the label side.

---

## 2. Title & Abstract (updated)

**Working titles:**
1. *"Letting the Ensemble Speak: Conformer-Derived Input Uncertainty in
   Variational Gaussian Process Regression for Molecular Properties"*
2. *"Uncertain by Nature: Propagating Conformer Ensemble Variance through
   Sparse Variational GPs on FreeSolv"*
3. *"Do Molecules Know Their Own Uncertainty? Conformer Ensembles as Input
   Noise for GPR"*

**Abstract sketch (draft):**

> Gaussian Process Regression (GPR) is attractive for small molecular
> datasets because it returns calibrated predictive distributions, but
> standard formulations treat each molecule as a single fixed point in
> descriptor space. In reality, a molecule at finite temperature is a
> Boltzmann ensemble of conformers, so any conformer-dependent descriptor is
> better described as a distribution than a point. We generate
> temperature-conditioned conformer ensembles for the FreeSolv hydration
> free energy dataset and use their empirical statistics to define a
> per-molecule input distribution \((\mu_i,\sigma_i)\). We train a sparse
> variational GP (inducing points, Cholesky variational distribution) using
> stochastic variational inference in which the model input is resampled
> from \(\mathcal{N}(\mu_i,\sigma_i)\) at every optimization step, so that
   the ELBO is an unbiased stochastic estimate over the conformer-derived
   input distribution. We compare this against an otherwise identical model
   trained on fixed mean-conformer inputs, evaluating both point-prediction
   accuracy and predictive calibration (NLPD, ENCE, coverage). We show
   [X: to be filled in after experiments] and analyze which molecules
   (by conformational flexibility) benefit most from explicit input-noise
   propagation.

---

## 3. Related Work (updated)

| Axis | Relevant prior work | Positioning |
|---|---|---|
| **Uncertain-input GPs via variational/stochastic training** | The GPyTorch SVI-with-resampled-inputs formalism you're building on; conceptually adjacent to Bayesian GP-LVM-style variational treatments of uncertain/latent inputs (Titsias & Lawrence, 2010) and to "Variational GP Dynamical Systems"-style work that marginalizes uncertain inputs variationally rather than via Taylor-expansion | You are the first (to your knowledge) to source \((\mu_i,\sigma_i)\) from a genuine physical simulation (a conformer ensemble) rather than an assumed/fit sensor-noise model |
| **Analytic input-noise propagation (contrast case)** | McHutchon & Rasmussen (2011), *"Gaussian Process Training with Input Noise"* (NIGP); Girard, Rasmussen, Quiñonero-Candela, Murray-Smith (2003) | Worth one paragraph contrasting: NIGP linearizes and re-fits iteratively (local, deterministic propagation); the SVI/resampling approach is exact-in-expectation given enough stochastic-gradient steps, at the cost of noisier optimization — a genuine methodological trade-off worth discussing, not just citing |
| **Uncertainty benchmarking on molecular property prediction** | Hirschfeld, Swanson, Yang, Barzilay, Coley (2020), JCIM; Scalia, Grambow, Pernici, Li, Green (2020), JCIM | Both treat noise/calibration as purely statistical; neither sources uncertainty from an explicit conformer ensemble — this is your clearest gap |
| **Molecular GPR / kernels** | GP libraries with Tanimoto/graph/string kernels for molecules (e.g. GAUCHE); SOAP/GAP-kernel GPR in materials (Deringer, Csányi et al.) | Representation-level tooling you'll likely reuse; none combine this with conformer-ensemble-derived input distributions |
| **FreeSolv-specific ML** | Mobley & Guthrie (2014) dataset paper; Vermeire & Green (2021) transfer-learning MPNN | Accuracy benchmarks to compare against; uncertainty, when reported, is ensemble-of-NN-based, not conformer-based |

**Verify all of the above (titles/years/venues) before citing — recalled from memory.**

---

## 4. Technical Concept

### 4.1 Conformer Ensemble Generation — detailed workflow

**Pipeline (applies to either tier below):**

1. **Parse & standardize.** Start from FreeSolv's canonical SMILES. Canonicalize,
   add explicit hydrogens, and respect any defined stereocenters when
   embedding — don't let the embedder silently pick an arbitrary
   enantiomer/diastereomer.
2. **Determine how many conformers to attempt** (see below — this should be
   adaptive per molecule, not a fixed global number).
3. **Generate an initial pool of 3D embeddings.**
4. **Optimize each embedded structure** with a force field or semiempirical
   method (see tiers below).
5. **Deduplicate** near-identical conformers by heavy-atom RMSD after
   alignment (threshold ~0.5–1.0 Å is standard).
6. **Apply an energy window filter** relative to the lowest-energy conformer
   found, and discard anything outside it — see the justification below.
7. **Compute Boltzmann weights** over the surviving, deduplicated set:
   \(w_k = \exp(-E_k/RT)/\sum_j \exp(-E_j/RT)\), \(T=298.15\) K to match
   FreeSolv's reference conditions.
8. **QC pass**: check that ensemble size correlates sensibly with rotatable
   bond count, spot-check a handful of molecules visually, and flag any
   molecule where embedding failed or produced a degenerate (single-point)
   ensemble.

**How many conformers per molecule (adaptive, not fixed):**

The number of *initial embedding attempts* should scale with molecular
flexibility — generating a fixed number (e.g. always 50) either wastes
compute on rigid molecules (most of FreeSolv's small solvents/simple
organics) or badly undersamples flexible ones. A commonly used heuristic
(from RDKit-conformer-generation benchmarking work, e.g. Ebejer, Morris &
Deane, 2012) scales attempts with the number of rotatable bonds:

| Rotatable bonds | Initial embedding attempts |
|---|---|
| 0–3 | ~50 |
| 4–6 | ~100 |
| 7–9 | ~200 |
| ≥10 | ~300 |

This is a starting point to calibrate, not a law — treat it as the *initial
pool size* before deduplication and energy-window filtering collapse it down
to the physically meaningful set. For FreeSolv specifically (mostly small,
often rigid-to-moderately-flexible molecules — many solvents, simple
drug-like fragments), expect the **final, filtered ensemble** to be small
for a large fraction of molecules — sometimes a single dominant conformer
for very rigid or symmetric molecules — and only reach a few dozen
meaningfully distinct, non-negligible-population conformers for the most
flexible cases (long alkyl chains, multiple rotatable bonds). This is
expected and fine — the whole point of goal 4 (studying which molecules
benefit from input-uncertainty propagation) depends on this real variation
in ensemble size/spread across the dataset.

**Why an energy window filter, and what value to use:** at \(T=298.15\) K,
\(RT \approx 0.593\) kcal/mol. A conformer 5 kcal/mol above the global
minimum contributes \(\exp(-5/0.593) \approx 3\times10^{-4}\) of the
minimum's Boltzmann weight — negligible. An energy window of roughly
3–4 kcal/mol above the lowest-energy conformer (≈5–7 \(RT\)) captures
>99% of the Boltzmann-weighted population in almost all practical cases,
and keeps the ensemble computationally manageable. Discarding
higher-energy conformers outright (rather than just down-weighting them) is
standard practice and defensible — their contribution to both \(\mu\) and
\(\sigma\) is vanishingly small.

**Two tiers, plus a recommended middle ground:**

- **Fast/practical tier:** RDKit `ETKDGv3` embedding (adaptive attempts per
  the table above) → MMFF94s optimization → Boltzmann-weight by MMFF
  energy. Cheap enough to run the entire FreeSolv set (~642 molecules) on a
  single CPU core in well under an hour. Good default if compute is
  constrained, but MMFF energies are a fairly crude energetic ranking.
- **Rigorous tier:** CREST (GFN2-xTB-based conformer-rotamer ensemble
  sampling via metadynamics) — automatically explores conformer space and
  outputs an energy-ranked ensemble with populations already computed.
  Substantially more defensible energetics, but more expensive: expect
  roughly minutes to tens of minutes per molecule depending on size/
  flexibility, so the full dataset needs an embarrassingly-parallel batch
  job (straightforward — molecules are independent) rather than serial
  execution, likely on the order of hours-to-a-few-days of wall-clock time
  depending on how many cores you can throw at it.
- **Recommended pragmatic middle ground:** generate the initial conformer
  pool cheaply with RDKit/MMFF94s (fast geometry sampling), then **rescore**
  each surviving conformer's energy with a single-point GFN2-xTB
  calculation before computing Boltzmann weights. This captures most of the
  energetic quality of the rigorous tier without paying for a full
  metadynamics search on every molecule, and is a common "cheap sampling +
  higher-level rescoring" pattern in computational conformer-search
  pipelines.

**A solvation-context nuance worth building in deliberately:** the property
you're predicting (hydration free energy) concerns the molecule *in water*,
not in vacuum. Gas-phase and aqueous-phase conformer populations can differ
meaningfully, especially for polar or flexible molecules where
intramolecular H-bonds that stabilize a gas-phase conformer may be
disfavored once the molecule is solvated. If using `xtb`/CREST, this is a
one-flag change (`--alpb water` or `--gbsa water`) to rank conformers by an
implicit-solvent energy rather than a vacuum one — worth doing by default
here rather than as an afterthought, since it's directly relevant to the
property being modeled and costs essentially nothing extra. Note this
explicitly in the paper as a deliberate methodological choice: your
conformer ensemble represents *the molecule as it behaves in water*, not
in vacuum, which is the physically appropriate reference state for a
hydration free energy task.

**A caveat worth stating explicitly in the paper:** this conformer ensemble
is your own independently-generated one (implicit-solvent, force-field or
semiempirical), distinct from whatever conformational sampling happened
inside FreeSolv's original explicit-solvent alchemical MD free energy
calculations. You're not trying to reproduce their simulation's internal
sampling — you're generating an independent, physically motivated estimate
of "how much does this molecule's structure vary at room temperature,"
which is then used purely to build \((\mu_i,\sigma_i)\).

### 4.2 From Ensemble to Per-Molecule Input Distribution \((\mu_i, \sigma_i)\)

This is now the central data product of the paper — everything downstream
consumes it directly.

- Split the representation into:
  - **Conformer-invariant block** \(x_{2D}\) (e.g. Morgan fingerprint / graph
    features): fixed, no distribution needed.
  - **Conformer-sensitive block** \(x_{3D}\) (dipole moment, radius of
    gyration, SASA, polarizability, or a 3D descriptor set): this is where
    the ensemble matters.
- For each molecule \(i\) and each dimension \(d\) of \(x_{3D}\), compute the
  Boltzmann-weighted mean and standard deviation across the ensemble:
  \(\mu_{i,d} = \sum_k w_k\, x_{3D,k,d}\), \(\sigma_{i,d}^2 = \sum_k w_k (x_{3D,k,d} - \mu_{i,d})^2\).
- This gives a **diagonal** Gaussian input distribution per molecule — matching
  the tutorial's per-dimension independent-Normal treatment (it does not
  model cross-descriptor covariance; that's a reasonable and defensible
  simplification worth stating explicitly, and a natural "future work" hook
  if you want an exact-covariance extension later).
- Concatenate the fixed \(x_{2D}\) block (zero variance) with the stochastic
  \(x_{3D}\) block (per-molecule \(\mu,\sigma\)) to form the full model input.

### 4.3 Model & Training (directly following the GPyTorch formalism)

- **Model:** `ApproximateGP` with a `VariationalStrategy` over learned
  inducing points, `CholeskyVariationalDistribution`, `ScaleKernel(RBFKernel())`
  or a Matérn kernel — matching the tutorial's structure. Likelihood:
  standard `GaussianLikelihood` (homoscedastic, single learned noise
  variance — no heteroscedastic component).
- **Training loop:** at every iteration, draw
  `x_sample = torch.distributions.Normal(mu, sigma).rsample()` for the
  stochastic block (concatenated with the fixed block), forward through the
  model, compute `-VariationalELBO(...)`, backprop, step. This is a direct
  transplant of the tutorial's training loop onto molecular descriptors.
- **Note on why variational/sparse, not exact GP:** the tutorial uses
  `ApproximateGP` because SVI naturally supports the per-step resampling
  trick; it is *not* motivated by needing scalability here (FreeSolv has
  ~642 points, well within exact-GP range). Because of this, a clean
  companion experiment is: **can the same resampling trick be applied to an
  `ExactGP`**, recomputing the exact marginal log-likelihood each step on
  the full (small) dataset with resampled inputs, instead of paying the
  inducing-point approximation cost? This isolates "cost of the sparse
  approximation" from "benefit of input-uncertainty propagation" — worth
  running as a robustness/ablation check (see 4.5) even though it isn't in
  the reference tutorial.

### 4.4 Prediction with Uncertain Test Inputs

The tutorial's toy example evaluates on deterministic test points; your test
molecules also have their own conformer ensembles, so this needs an explicit
procedure (this is a genuine extension you'll want to describe carefully in
Methods):

- For each test molecule, draw \(M\) samples \(x^{(m)} \sim \mathcal{N}(\mu_{test}, \sigma_{test})\).
- Get the predictive distribution for each sample from the trained
  variational GP + likelihood: \(p(y \mid x^{(m)})\), each Gaussian with mean
  \(m_m\) and variance \(v_m\).
- Combine via the law of total variance to get one final predictive
  Gaussian (or a Gaussian-mixture if you want to keep it non-Gaussian):
  \(\bar{m} = \frac{1}{M}\sum_m m_m\), \(\bar{v} = \frac{1}{M}\sum_m v_m + \frac{1}{M}\sum_m (m_m - \bar{m})^2\).
  The second term is exactly the *extra* variance contributed by input
  uncertainty — report it separately as a diagnostic, since it's the most
  direct, interpretable number for "how much did conformer uncertainty add."

### 4.5 Ablation Design (Goal: isolate the effect of input uncertainty)

Keep architecture identical across variants so the comparison is clean —
only the input-handling differs:

| Variant | Input at train time | Input at test time |
|---|---|---|
| A: Deterministic baseline | fixed \(\mu_i\) (ensemble mean only, no resampling) | fixed \(\mu_{test}\) |
| B: Uncertain-input SVGP (core model) | resampled \(x^{(t)}\sim\mathcal{N}(\mu_i,\sigma_i)\) each step | MC-averaged over \(M\) test samples (4.4) |
| C (robustness check): Uncertain-input **exact** GP | resampled \(x^{(t)}\sim\mathcal{N}(\mu_i,\sigma_i)\) each step, exact MLL, no inducing points | MC-averaged, same as B |

A vs. B isolates the effect of input-uncertainty propagation given identical
(sparse variational) machinery — this is your headline ablation. B vs. C
isolates the cost of the sparse approximation itself, which matters for
interpreting whether any observed effect is really about input uncertainty
or partly an artifact of the inducing-point approximation. Stratify all
comparisons by a flexibility proxy (rotatable bond count, or ensemble
spread) to test whether benefits concentrate in conformationally flexible
molecules.

---

## 5. Evaluation Protocol — what to emphasize and why

The single most important framing decision in the whole paper: **accuracy
metrics (RMSE/MAE/R²) are a sanity check, not the headline result.** Variant
A and Variant B share the same kernel, same inducing points, same
likelihood — the input-uncertainty propagation mainly reshapes the
*predictive variance*, not the *predictive mean*. If you lead with "our
uncertain-input model achieves lower RMSE," you're setting up the wrong
comparison and inviting a reviewer to (correctly) ask why a method designed
to improve calibration is being sold on accuracy. Lead with calibration;
report accuracy mainly to show it *didn't get worse*.

### 5.1 Accuracy metrics (secondary — a "did nothing break" check)

- RMSE, MAE, R² against published FreeSolv benchmarks, and — more
  importantly for your internal comparison — between Variants A and B
  themselves. Expect these to be close; a large accuracy gap in either
  direction is worth investigating (e.g. if B is *much* worse, the
  resampling noise may be destabilizing optimization; if B is *much*
  better, check that isn't just underfitting-regularization from the
  injected noise acting like data augmentation — worth explicitly
  discussing either way, since it's a real possible confound of stochastic
  input resampling, not a modeling failure).

### 5.2 Calibration metrics (primary — this is the actual contribution)

These are **proper scoring rules** or calibration diagnostics — they
reward a model for being both accurate *and* appropriately (not over- or
under-) confident, which RMSE/MAE cannot assess at all. GPyTorch ships
several of these directly (`gpytorch.metrics`), so use them as-is for
correctness and reproducibility rather than reimplementing:

- **NLPD (Negative Log Predictive Density)** — the primary headline metric.
  Penalizes a confident-and-wrong prediction much more than an
  appropriately-uncertain-and-wrong one, so it directly rewards good
  calibration, not just low error. `gpytorch.metrics.negative_log_predictive_density`.
- **MSLL (Mean Standardized Log Loss)** — NLPD normalized against a trivial
  baseline that just predicts the training-set mean and variance for every
  point. Useful because it's more interpretable across different
  datasets/scales than raw NLPD, and because a model that's *worse* than
  the trivial baseline (MSLL > 0) is an immediate red flag.
  `gpytorch.metrics.mean_standardized_log_loss`.
- **ENCE (Expected Normalized Calibration Error)** — bins test points by
  predicted uncertainty and compares the empirical RMSE within each bin to
  the predicted standard deviation for that bin. This is the metric that
  most directly answers "when the model says it's uncertain, is it
  actually more often wrong by that much" — arguably the most intuitive
  calibration metric to put in a figure (as a scatter/line plot of
  predicted-vs-empirical error per bin).
- **Quantile coverage error** — e.g. does the nominal 95% interval actually
  contain ~95% of test points? `gpytorch.metrics.quantile_coverage_error`
  computes this directly; report at multiple nominal levels (50%, 80%,
  95%) rather than just one, and plot a full reliability diagram
  (nominal vs. empirical coverage across all quantile levels) — a single
  coverage number can look fine while the underlying curve is
  systematically off in a way a plot would reveal immediately.
- **Sharpness** — the average predicted variance, reported *conditional on*
  the model being reasonably calibrated. A calibrated-but-sharp
  (confident) model is strictly better than a calibrated-but-diffuse one;
  don't report sharpness in isolation, since a model can trivially get
  "sharp" by being overconfident and wrong. Pair it with ENCE/coverage.
- **Spearman correlation between predicted uncertainty and absolute
  error** — weaker than the above (only checks *ranking*, not calibrated
  magnitude), but easy to compute and intuitive to report: does the model
  at least know *which* of its predictions to trust less, even before
  asking whether the numbers are exactly right.
- **(Optional) CRPS (Continuous Ranked Probability Score)** — another
  proper scoring rule, closed-form for Gaussian predictive distributions;
  worth including only if you want a second proper-scoring-rule metric
  alongside NLPD to show the conclusion isn't an artifact of one
  particular scoring choice.

### 5.3 Metrics specific to the input-uncertainty contribution (this is where the paper's actual novelty gets measured)

- **The "extra variance from input uncertainty" term** from the
  law-of-total-variance decomposition in §4.4
  (\(\frac{1}{M}\sum_m (m_m - \bar m)^2\)) — report its distribution across
  the test set (not just a single average), and **correlate it against a
  flexibility proxy** (rotatable bond count, or the size/spread of the
  molecule's filtered conformer ensemble from §4.1). This is the most
  direct, mechanistic evidence for "the model is using conformational
  information sensibly" — if this term doesn't correlate with flexibility
  at all, that's an important (if negative) finding to report honestly,
  since it would mean the propagation mechanism isn't doing physically
  meaningful work even if calibration numbers move.
- **Fraction of total predictive variance attributable to input
  uncertainty** vs. the GP's own epistemic variance and the learned
  likelihood noise — a variance-partitioning view, reported as an average
  and as a per-molecule breakdown. This tells the calibration story in
  physically interpretable units rather than just an abstract score.
- **Stratified calibration**: recompute NLPD/ENCE/coverage separately for
  a "flexible" subset vs. a "rigid" subset of molecules (split by median
  rotatable-bond count, or by filtered-ensemble size from §4.1). The
  hypothesis worth explicitly testing — and structuring a whole subsection
  and figure around — is that **Variant B's calibration advantage (if any)
  concentrates in the flexible subset** and is negligible or absent for
  rigid molecules where \(\sigma_i \approx 0\) anyway (in which case A and B
  should be nearly identical by construction — a useful internal
  consistency check).

### 5.4 Statistical robustness (don't skip this — FreeSolv is small enough that noise matters)

- Run **multiple random seeds** for both Variant A and Variant B, and
  report mean ± std (or a proper paired test) for every metric above,
  across both random and scaffold splits. FreeSolv's modest size (~642
  molecules) means metric differences between A and B can easily be
  smaller than run-to-run optimization noise, especially since B's
  training loop introduces additional stochasticity (the resampling
  itself) on top of ordinary SGD noise shared with A.
- Use a **paired test** (e.g. paired t-test or Wilcoxon signed-rank across
  matched seeds/splits) rather than comparing single-run numbers — this is
  the difference between "B looks better" and "B is better" in a paper
  that will be read by people who benchmark FreeSolv for a living.
- Sanity-check optimization itself: plot ELBO/training-loss curves for A
  vs. B — if B's stochastic resampling is destabilizing convergence
  (rather than just adding a benign amount of gradient noise), that's a
  confound you want to catch and address (e.g. via learning-rate tuning
  or more MC samples per step) before trusting any downstream calibration
  comparison.

### 5.5 Qualitative diagnostics (cheap, and often more persuasive to readers than another table)

- **Reliability diagrams** for A vs. B, side by side.
- **Predicted-uncertainty vs. actual-error scatter/binned plots**, A vs. B.
- **Two or three case-study molecules** — one rigid, one highly flexible —
  showing the predictive distribution under A vs. B explicitly. This is
  often the single most convincing figure in a paper like this: it makes
  the abstract "input uncertainty propagation" claim visually concrete for
  one rigid molecule (A ≈ B, as expected) and one flexible molecule (B
  visibly wider/better-calibrated than A).

### 5.6 External baselines

- Compare against reported deep-ensemble / MC-dropout calibration numbers
  from Hirschfeld et al. (2020) and Scalia et al. (2020) where feasible —
  ideally on matching splits, otherwise clearly caveat that the comparison
  is cross-study rather than head-to-head.

---

## 6. Roadmap

**Phase 0 — Scoping & Literature (1–1.5 weeks)**
- Confirm conformer tier (RDKit/MMFF vs. CREST) and 3D descriptor set.
- Read/adapt the GPyTorch tutorial end-to-end on a toy 1D case before
  touching molecular data, to get the training loop, inducing-point count,
  and ELBO convergence behavior calibrated.
- Deliverable: working toy reproduction of the tutorial + descriptor-set
  decision memo.

**Phase 1 — Conformer Ensembles & \((\mu,\sigma)\) Pipeline (2.5–3.5 weeks)**
- Build ensemble generation for all FreeSolv molecules; compute
  Boltzmann-weighted \(\mu_{i,d}, \sigma_{i,d}\) per conformer-sensitive
  descriptor dimension.
- Sanity checks: does \(\sigma\) correlate with rotatable-bond count? Spot
  check a handful of molecules (e.g. one rigid, one floppy) by hand/plot.
- Deliverable: reproducible, cached dataset of (molecule → \(x_{2D}\) →
  \(\mu_{3D}, \sigma_{3D}\)).

**Phase 2 — Baseline Deterministic SVGP (2 weeks)**
- Implement Variant A (deterministic-input sparse variational GP) exactly
  per the GPyTorch architecture, no resampling.
- Validate against a published FreeSolv RMSE ballpark before adding
  complexity — this is your correctness checkpoint.
- Deliverable: Model A trained, evaluated, RMSE in literature range.

**Phase 3 — Uncertain-Input SVGP (3–4 weeks)**
- Implement the resampling training loop (Variant B) exactly following the
  tutorial's structure, adapted to the concatenated
  fixed-block/stochastic-block input.
- Implement the multi-sample test-time prediction + variance decomposition
  (4.4).
- Deliverable: Model B trained end-to-end with working MC prediction.

**Phase 4 — Exact-GP Robustness Check (1.5–2 weeks, can run in parallel with Phase 3 once B is working)**
- Implement Variant C (resampling + exact GP, no inducing points) as a
  robustness/ablation check, given FreeSolv's small size makes this
  tractable.
- Deliverable: Model C trained, compared against B.

**Phase 5 — Full Ablation Grid, Calibration Analysis, Stratification (3–4 weeks)**
- Run A vs. B vs. C across random + scaffold splits, multiple seeds.
- Compute all calibration metrics via `gpytorch.metrics`; build reliability
  diagrams.
- Stratify by flexibility proxy; test "does input uncertainty help more for
  floppy molecules" hypothesis.
- Compare against external UQ benchmark numbers (Hirschfeld/Scalia).
- Deliverable: full results tables + figures, draft Results section.

**Phase 6 — Writing & Internal Review (3–4 weeks)**
- Draft Introduction, Methods, Results, Discussion; get feedback; revise.
- Deliverable: full manuscript draft + SI.

**Phase 7 — Submission & Response Cycle (ongoing)**

**Total estimated timeline:** ~17–21 weeks of focused work to a first
submittable draft — slightly shorter than v1, mainly because dropping the
heteroscedastic-likelihood development (old Phase 3) removes a few weeks of
work, offset somewhat by the new exact-GP robustness check and the
non-trivial test-time MC-prediction procedure.

---

## 7. Target Venues (unchanged)

- **Journal of Chemical Information and Modeling (JCIM)** — primary target,
  directly extends the Hirschfeld/Scalia UQ-benchmark lineage.
- **Digital Discovery (RSC)** — good fit, open access.
- **Machine Learning: Science and Technology (MLST)** — if you want to lean
  more into the GP-methodology angle (variational uncertain-input training)
  as the primary contribution.
- Workshop pre-print (NeurIPS/ICML AI4Science-style) as an early checkpoint
  before the full journal submission.

---

## 8. Open Decisions to Pin Down Early

1. **Which descriptors go in the stochastic (conformer-sensitive) block vs.
   the fixed block** — this determines both how much signal comes from
   input-uncertainty propagation and how large \(\sigma_i\) typically is.
2. **Number of inducing points and their initialization** — with only ~642
   molecules, inducing-point count should probably be a real fraction of the
   dataset (e.g. 50–150), not the "few dozen for millions of points" regime
   the sparse-GP literature usually targets; worth a small sweep.
3. **Number of MC samples \(M\) at test time** (4.4) — trade-off between
   variance-of-the-estimate and compute; 20–50 is a reasonable starting
   range to sweep.
4. **Whether to include Variant C (exact-GP robustness check)** at all if
   compute/time is tight — it strengthens the paper's internal validity
   claim but isn't strictly required to answer the core research question.
5. **Conformer generation tier** (RDKit/MMFF vs. CREST) — same trade-off as
   before: compute budget vs. strength of the "physically grounded" claim.

---

*This is a living document. The GPyTorch formalism prescribes the training
mechanics precisely; the main remaining design freedom is in Section 4.2
(which descriptors, how the ensemble statistics are computed) and Section
4.4 (the test-time MC prediction procedure, which is your own extension
beyond the tutorial and deserves careful description in Methods).*
