"""
Utilities to load descriptor distribution data (mu, sigma) and to train/test
Exact GPR models in GPyTorch.

Provided models:
- Uncertain-input GP: Gaussian Symmetrized-KL kernel over diagonal-Gaussian inputs (uses mu and sigma).
- Deterministic GP: standard RBF kernel over descriptor means only (uses mu only).

This module is intentionally lightweight and makes only minimal assumptions:
- Input file is an .npz produced by compute_coulomb_descriptors.py containing
  arrays: ids (int), mu (N x D), sigma (N x D), plus metadata.
- Labels can be provided as a CSV with columns [SMILES,label] (FreeSolv
  convention) or as a numpy array aligned by ids.

Two primary entry points for each setting are exposed:
- Uncertain: train_gpr_uncertain(mu, sigma, y, config) and evaluate_gpr_uncertain(...)
- Deterministic: train_gpr_deterministic(mu, y, config) and evaluate_gpr_deterministic(...)

Note on input-uncertainty handling (uncertain mode):
We use a closed-form kernel over Gaussian inputs based on the
symmetrized KL divergence between diagonal Gaussians, transformed by an
RBF-style exponent. This avoids Monte Carlo input sampling while correctly
accounting for per-dimension uncertainty magnitudes (as given in the following resource:
https://docs.gpytorch.ai/en/v1.15.2/examples/01_Exact_GPs/GP_Regression_DistributionalKernel.html)
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional, Tuple
from hydra import initialize, compose
from omegaconf import DictConfig

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, GaussianSymmetrizedKLKernel, PolynomialKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.preprocess import VarianceFloor
from src.gaussian_expected_polynomial_kernel import GaussianExpectedPolynomialKernel


def load_descriptors_npz(path: str) -> Dict[str, np.ndarray]:
    """Load ids, mu, sigma (+metadata) from an .npz produced by
    compute_coulomb_descriptors.py.

    Returns a dict with keys: ids (np.ndarray[int]), mu (np.ndarray[float64]),
    sigma (np.ndarray[float64]), and any extra metadata keys present.
    """
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    # Normalize key names we care about
    required = ["ids", "mu", "sigma"]
    for k in required:
        if k not in out:
            raise KeyError(f"Missing '{k}' in {path}; got keys {list(out.keys())}")
    return out


def load_freesolv_labels(csv_path: str) -> np.ndarray:
    """Load labels from a FreeSolv CSV (columns: SMILES,label) into a numpy array.

    The returned array y is in the same row order as the CSV file. If your
    descriptor ids are integer indices (0..N-1), you can align via y[ids].
    """
    import csv

    ys = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if "label" not in reader.fieldnames:
            raise ValueError("CSV missing 'label' column")
        for row in reader:
            ys.append(float(row["label"]))
    return np.asarray(ys, dtype=np.float64)


@dataclasses.dataclass
class TrainConfig:
    # Optimization
    lr: float = 0.05
    epochs: int = 200
    weight_decay: float = 0.0
    # Model
    use_ard: bool = False  # ARD lengthscale per descriptor dim
    jitter: float = 1e-6
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    # Logging
    verbose: bool = True


class ExactGPRWithSKL(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood, degree: int = 0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        base = GaussianSymmetrizedKLKernel() if degree == 0 else GaussianExpectedPolynomialKernel(power=degree)
        self.covar_module = ScaleKernel(base)

    def forward(self, x: torch.Tensor):
        # Build covariance first to read off its batch shape and event size (N x N)
        covar_x = self.covar_module(x)
        # Ensure the mean matches the covariance's batch shape to avoid broadcasting issues.
        # covar_x.shape = batch_shape + (N, N) -> mean should be batch_shape + (N,)
        cov_shape = covar_x.shape
        batch_shape = cov_shape[:-2]
        N = cov_shape[-1]
        mean_x = torch.zeros(*batch_shape, N, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPRStandard(gpytorch.models.ExactGP):
    """Deterministic-input Exact GP with RBF kernel over descriptor means only."""
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood, use_ard: bool = True, degree: int = 0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        ard = train_x.shape[-1] if use_ard else None
        base = RBFKernel(ard_num_dims=ard) if degree == 0 else PolynomialKernel(power=degree)
        self.covar_module = ScaleKernel(base)

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _sample_inputs(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mu)
    return mu + sigma * eps


def train_gpr_uncertain(mu: torch.Tensor, var: torch.Tensor, y: np.ndarray, cfg: DictConfig):
    """Train an Exact GP using the Gaussian symmetrized-KL kernel.

    Args:
      mu, var: numpy arrays (N, D) describing input distributions N(mu, diag(var)).
      y: numpy array (N,) of targets aligned with rows of mu/sigma.
      cfg: TrainConfig with training hyperparameters.

    Returns: (model, likelihood)
    """
    if cfg is None:
        cfg = TrainConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    N, D = mu.shape
    X_mu = mu
    y_t = torch.as_tensor(y, dtype=dtype, device=device)

    # Defensive floor: do NOT rely solely on upstream preprocessing (VarianceFloor /
    # DistributionalInputStandardizer) to guarantee var > 0. log() of a non-positive
    # variance produces NaN/-inf, and a NaN anywhere in X_concat makes ExactGP's
    # `torch.equal(train_input, input)` sanity check fail even when the identical tensor
    # object is passed twice -- because NaN != NaN under IEEE float rules. This surfaces
    # as the confusing "You must train on the training inputs!" error rather than a NaN
    # error, which is what sent us on a long chase last time this happened.
    var_floor = 1e-30  # far below any real variance scale we've seen in this project;
                        # only meant to catch var <= 0 slipping through, not to define scale
    n_bad = int((var <= 0).sum().item())
    if n_bad > 0:
        print(f"[warn] {n_bad} variance entries were <= 0 before flooring; clamping to {var_floor}")
    var = var.clamp_min(var_floor)

    var_term = var @ var.transpose(-1, -2)
    mean_term = mu @ mu.transpose(-1, -2)
    print("var-term range:", var_term.min().item(), var_term.max().item())
    print("mean-term range:", mean_term.min().item(), mean_term.max().item())

    # Concatenate [mu | log(var)] as required by the SKL kernel.
    # GaussianSymmetrizedKLKernel expects a FLAT (N, 2*D) tensor: first D columns are
    # means, last D columns are log-variances. (It does NOT want a stacked (N, D, 2)
    # tensor -- that adds a spurious extra dimension that gpytorch treats as a batch
    # dimension, which is what was causing the shape-mismatch crash.)
    X_concat = torch.cat((X_mu, var.log()), dim=-1)  # (N, 2*D)
    if not torch.isfinite(X_concat).all():
        raise ValueError(
            "X_concat contains NaN/Inf after building [mu | log(var)]. Check that mu "
            "has no NaNs and that var is finite before it reaches train_gpr_uncertain "
            "(this floor only guards var <= 0, not NaN already present in mu or var)."
        )
    likelihood = GaussianLikelihood().to(device=device, dtype=dtype)


    model = ExactGPRWithSKL(X_concat, y_t, likelihood, degree=cfg.training.kernel_degree).to(device=device, dtype=dtype)


    model.train()
    likelihood.train()

    # Build optimizer while avoiding duplicate parameters across model and likelihood
    import itertools
    unique_params = list({id(p): p for p in itertools.chain(model.parameters(), likelihood.parameters())}.values())
    optimizer = torch.optim.Adam(unique_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(cfg.training.epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(X_concat)
        # Ensure target matches the batch shape of the model output, if any (e.g., when
        # the distributional axis is treated as a batch dimension by the kernel).
        y_target = y_t
        if output.mean.dim() > 1:
            # Expand y to output.batch_shape + (N,)
            batch_shape = output.mean.shape[:-1]
            y_target = y_t.view((1,) * len(batch_shape) + y_t.shape).expand(batch_shape + y_t.shape)
        loss = -mll(output, y_target)
        loss.backward()
        optimizer.step()
        if (epoch % max(1, cfg.training.epochs // 10) == 0 or epoch == cfg.training.epochs - 1):
            print(f"[train] epoch {epoch+1}/{cfg.training.epochs}  loss={float(loss.detach().cpu()):.4f}")

    return model, likelihood


@torch.no_grad()
def predict_gpr_uncertain(model: ExactGPRWithSKL, likelihood: GaussianLikelihood, mu: np.ndarray, var: np.ndarray, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic prediction with the SKL kernel.

    Note: mc_samples and batch_size are accepted for backward compatibility but
    are ignored since the kernel already integrates input uncertainty.
    """
    model.eval()
    likelihood.eval()

    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype

    X_mu = torch.as_tensor(mu, dtype=dtype, device=device)
    # Match training representation: flat (N, 2*D) tensor, means then log-variances.
    X_concat = torch.cat((X_mu, var.log()), dim=-1)  # (N, 2*D)

    out = likelihood(model(X_concat))
    mean = out.mean
    pred_var = out.variance
    # If the kernel produced a batched output (e.g., because the distributional axis
    # was treated as a batch), average predictions across batch dims to return (N,)
    while mean.dim() > 1:
        mean = mean.mean(dim=0)
    while pred_var.dim() > 1:
        pred_var = pred_var.mean(dim=0)
    return mean.detach().cpu().numpy(), pred_var.detach().cpu().numpy()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nll_gaussian(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    # Negative log likelihood under predictive Gaussian N(mean, var)
    eps = 1e-9
    var = np.maximum(var, eps)
    return float(0.5 * np.mean(np.log(2 * math.pi * var) + (y_true - mean) ** 2 / var))


def evaluate_gpr_uncertain(model: ExactGPRWithSKL, likelihood: GaussianLikelihood, mu: np.ndarray, var: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
    mean, pred_var = predict_gpr_uncertain(model, likelihood, mu, var)
    metrics: Dict[str, float] = {}
    if y is not None:
        metrics["rmse"] = rmse(y, mean)
        metrics["mae"] = mae(y, mean)
        metrics["nll"] = nll_gaussian(y, mean, pred_var)
    metrics["mean_var"] = float(np.mean(pred_var))

    print(pred_var)
    return metrics


# ===== Deterministic-input GP (no input uncertainty) =====

def train_gpr_deterministic(mu: np.ndarray, y: np.ndarray, cfg: DictConfig):
    """Train an Exact GP on deterministic inputs using an RBF kernel over mu.

    Args:
      mu: numpy array (N, D) of descriptor means (deterministic inputs).
      y: numpy array (N,) of targets aligned with rows of mu.
      cfg: TrainConfig with training hyperparameters.

    Returns: (model, likelihood)
    """
    if cfg is None:
        cfg = TrainConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    X = torch.as_tensor(mu, dtype=dtype, device=device)
    y_t = torch.as_tensor(y, dtype=dtype, device=device)

    likelihood = GaussianLikelihood().to(device=device, dtype=dtype)
    model = ExactGPRStandard(X, y_t, likelihood, use_ard=cfg.training.use_ard, degree=cfg.training.kernel_degree).to(device=device, dtype=dtype)

    model.train()
    likelihood.train()

    # Build optimizer with unique parameter set to avoid duplicates across model and likelihood
    import itertools
    unique_params = list({id(p): p for p in itertools.chain(model.parameters(), likelihood.parameters())}.values())
    optimizer = torch.optim.Adam(unique_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(cfg.training.epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(X)
        loss = -mll(output, y_t)
        loss.backward()
        optimizer.step()
        # if (epoch % max(1, cfg.training.epochs // 10) == 0 or epoch == cfg.training.epochs - 1):
        #    print(f"[train-det] epoch {epoch+1}/{cfg.training.epochs}  loss={float(loss.detach().cpu()):.4f}")

    return model, likelihood


@torch.no_grad()
def predict_gpr_deterministic(model: ExactGPRStandard, likelihood: GaussianLikelihood, mu: np.ndarray, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    likelihood.eval()

    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype

    X = torch.as_tensor(mu, dtype=dtype, device=device)
    out = likelihood(model(X))
    mean = out.mean.detach().cpu().numpy()
    var = out.variance.detach().cpu().numpy()
    return mean, var


def evaluate_gpr_deterministic(model: ExactGPRStandard, likelihood: GaussianLikelihood, mu: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
    mean, pred_var = predict_gpr_deterministic(model, likelihood, mu)
    metrics: Dict[str, float] = {}
    if y is not None:
        metrics["rmse"] = rmse(y, mean)
        metrics["mae"] = mae(y, mean)
        metrics["nll"] = nll_gaussian(y, mean, pred_var)
    metrics["mean_var"] = float(np.mean(pred_var))

    # Print predictive variances
    # print(pred_var)

    return metrics


def get_predictive_uncertainty_uncertain(model: ExactGPRWithSKL, likelihood: GaussianLikelihood, mu: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return per-point predictive variance and its mean for uncertain-input GP.

    Args:
      model, likelihood: trained GP with SKL kernel.
      mu, sigma: test-set descriptor means and stddevs, shape (N, D).

    Returns:
      (var_per_point, mean_variance)
    """
    _, var = predict_gpr_uncertain(model, likelihood, mu, sigma)
    return var, float(np.mean(var))


def get_predictive_uncertainty_deterministic(model: ExactGPRStandard, likelihood: GaussianLikelihood, mu: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return per-point predictive variance and its mean for deterministic-input GP.

    Args:
      model, likelihood: trained standard GP.
      mu: test-set descriptor means, shape (N, D).

    Returns:
      (var_per_point, mean_variance)
    """
    _, var = predict_gpr_deterministic(model, likelihood, mu)
    return var, float(np.mean(var))


def kfold_cv_mae_gpr_uncertain(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, cfg: Optional[DictConfig] = None, k: int = 5, seed: int = 0) -> float:
    """Compute k-fold cross-validated MAE on the provided training set for the
    uncertain-input GP (SKL kernel).

    Args:
      mu: (N, D) descriptor means.
      sigma: (N, D) descriptor stddevs.
      y: (N,) targets aligned with rows of mu/sigma.
      cfg: training config; if None, defaults will be used.
      k: number of folds (>=2).
      seed: random seed for shuffling before creating folds.

    Returns:
      The average MAE across the k folds.
    """
    if cfg is None:
        cfg = TrainConfig()
    N = len(y)
    if k < 2:
        raise ValueError("k must be >= 2 for k-fold CV")
    if k > N:
        raise ValueError("k cannot exceed number of samples")

    # Create shuffled indices and contiguous folds
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    maes = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]) if k > 1 else folds[i]
        mu_tr, sg_tr, y_tr = mu[train_idx], sigma[train_idx], y[train_idx]
        mu_va, sg_va, y_va = mu[val_idx], sigma[val_idx], y[val_idx]

        model, likelihood = train_gpr_uncertain(mu_tr, sg_tr, y_tr, cfg)
        mean_va, _ = predict_gpr_uncertain(model, likelihood, mu_va, sg_va)
        maes.append(mae(y_va, mean_va))
    return float(np.mean(maes))


def kfold_cv_mae_gpr_deterministic(mu: np.ndarray, y: np.ndarray, cfg: Optional[DictConfig] = None, k: int = 5, seed: int = 0) -> float:
    """Compute k-fold cross-validated MAE on the provided training set for the
    deterministic-input GP (RBF over means).

    Args:
      mu: (N, D) descriptor means.
      y: (N,) targets aligned with rows of mu.
      cfg: training config; if None, defaults will be used.
      k: number of folds (>=2).
      seed: random seed for shuffling before creating folds.

    Returns:
      The average MAE across the k folds.
    """
    if cfg is None:
        cfg = TrainConfig()
    N = len(y)
    if k < 2:
        raise ValueError("k must be >= 2 for k-fold CV")
    if k > N:
        raise ValueError("k cannot exceed number of samples")

    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    maes = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]) if k > 1 else folds[i]
        mu_tr, y_tr = mu[train_idx], y[train_idx]
        mu_va, y_va = mu[val_idx], y[val_idx]

        model, likelihood = train_gpr_deterministic(mu_tr, y_tr, cfg)
        mean_va, _ = predict_gpr_deterministic(model, likelihood, mu_va)
        maes.append(mae(y_va, mean_va))
    return float(np.mean(maes))


def split_train_test_by_fraction(N: int, train_frac: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_train = int(round(train_frac * N))
    return idx[:n_train], idx[n_train:]


def demo_train_test(npz_path: str, csv_labels_path: Optional[str] = None, seed: int = 0) -> Dict[str, float]:
    """Small convenience routine to train/test a model from the descriptors .npz.

    If csv_labels_path is provided (e.g., data/freesolv.csv), labels are read
    and aligned via ids (assumed to be integer indices into the CSV row order).
    Otherwise, raises if labels cannot be obtained.
    """
    data = load_descriptors_npz(npz_path)
    mu = data["mu"]
    sigma = data["sigma"]
    ids = data["ids"].astype(int)

    if csv_labels_path is None:
        raise ValueError("csv_labels_path is required to obtain targets for training/testing")
    y_all = load_freesolv_labels(csv_labels_path)

    y = y_all[ids]

    train_idx, test_idx = split_train_test_by_fraction(len(ids), 0.8, seed)
    mu_tr, sg_tr, y_tr = mu[train_idx], sigma[train_idx], y[train_idx]
    mu_te, sg_te, y_te = mu[test_idx], sigma[test_idx], y[test_idx]

    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config")

    model, likelihood = train_gpr_uncertain(mu_tr, sg_tr, y_tr, cfg)

    metrics = evaluate_gpr_uncertain(model, likelihood, mu_te, sg_te, y_te)
    print({k: float(v) for k, v in metrics.items()})
    return metrics