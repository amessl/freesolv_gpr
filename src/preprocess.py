
"""
Variance floor for the distributional-kernel (GaussianSymmetrizedKLKernel)
uncertain-input GPR pipeline.

Background / why this exists:
Each molecule's input to the distributional kernel is represented as
(mean, log-variance) per descriptor dimension. A meaningful fraction of
FreeSolv molecules collapse to a single surviving conformer after ensemble
filtering (verified earlier in this project: e.g. benzene, ethanol,
acetaminophen-like all gave sigma == 0 exactly), which means variance == 0
for those molecules/dimensions. log(0) = -inf, and injecting -inf into the
kernel's input tensor corrupts training for every point the kernel touches,
not just the affected molecules.

This module fits a small positive floor (eps) on variance, from the
TRAINING set only, and applies it consistently everywhere (train, val,
test) via clamp-before-log. eps is chosen as a low percentile of the
*nonzero* variances actually observed in training, rather than an
arbitrary constant, so it stays scale-appropriate for whatever descriptor
you're using (Coulomb matrix, SOAP, etc.) without hand-tuning per dataset.

Usage:
    floor = VarianceFloor(percentile=1.0)
    logvar_train = floor.fit_transform(sigma_train)   # fits eps on train
    logvar_test = floor.transform(sigma_test)         # reuses the same eps

    # then build the kernel input, e.g.:
    train_x_distributional = torch.cat([mu_train, logvar_train], dim=-1)
"""

import numpy as np
import torch


class VarianceFloor:
    """Fits a data-driven variance floor (eps) from a training set's sigma
    values, then applies it consistently wherever needed (train, val, test).

    eps is set to a low percentile of the *nonzero* variances observed in
    training -- low enough not to distort genuine small variances, high
    enough that floored (exactly-rigid) molecules don't become extreme
    outliers on the log scale relative to the rest of the data. Fit once on
    the training set only; reusing that frozen value on val/test avoids
    leakage from those splits into a preprocessing choice.
    """

    def __init__(self, percentile: float = 50.0):
        self.percentile = percentile
        self.eps_ = None  # populated by .fit()

    def fit(self, variance: torch.Tensor, verbose: bool = True) -> "VarianceFloor":

        nonzero = variance[variance > 0]
        if nonzero.numel() == 0:
            raise ValueError(
                "No nonzero variances found in the training set -- check "
                "sigma computation upstream (e.g. did the ensemble-generation "
                "step actually produce multi-conformer molecules?) before "
                "picking a floor."
            )

        self.eps_ = float(np.percentile(nonzero.numpy(), self.percentile))

        if verbose:
            n_zero = int((variance == 0).sum())
            n_total = variance.numel()
            median_nonzero = float(np.median(nonzero.numpy()))
            print(f"[VarianceFloor.fit] {n_zero}/{n_total} entries "
                  f"({100 * n_zero / n_total:.1f}%) exactly zero in training data")
            print(f"[VarianceFloor.fit] nonzero variance range: "
                  f"[{nonzero.min():.3e}, {nonzero.max():.3e}]")
            print(f"[VarianceFloor.fit] eps = {self.eps_:.3e} "
                  f"({self.percentile}th percentile of nonzero variances)")
            gap = np.log(median_nonzero) - np.log(self.eps_)
            print(f"[VarianceFloor.fit] log(eps)={np.log(self.eps_):.2f} vs. "
                  f"log(median nonzero)={np.log(median_nonzero):.2f} "
                  f"(gap: {gap:.2f} -- if this is large, e.g. >6-8, consider "
                  f"a higher percentile so the floor isn't an extreme outlier)")
        return self

    def transform(self, variance: torch.Tensor) -> torch.Tensor:
        """Returns floored log-variance, ready to feed into the
        distributional kernel's input tensor. Always uses the eps fitted on
        the training set, regardless of which split `sigma` comes from."""
        if self.eps_ is None:
            raise RuntimeError("call .fit() on the training set before .transform()")
        variance_floored = torch.clamp(variance, min=self.eps_)
        return variance_floored

    def fit_transform(self, var_train: torch.Tensor) -> torch.Tensor:
        self.fit(var_train)
        return self.transform(var_train)


def diagnose_uncertain_input(mu: torch.Tensor, var: torch.Tensor):
    """Run this on your REAL (mu, sigma) tensors before training, to see
    your actual dataset's numbers -- not a synthetic stand-in."""
    print("mu:    any NaN?", torch.isnan(mu).any().item(),
          " any Inf?", torch.isinf(mu).any().item())
    print("sigma: any NaN?", torch.isnan(var).any().item(),
          " any Inf?", torch.isinf(var).any().item())
    n_fully_rigid = (var.sum(dim=1) == 0).sum().item()
    print(f"molecules with sigma==0 across ALL dims (fully rigid): "
          f"{n_fully_rigid} / {var.shape[0]}")
    logvar_raw = torch.log(var)  # WITHOUT any floor -- see what breaks
    n_inf = torch.isinf(logvar_raw).sum().item()
    print(f"raw log(sigma^2) without a floor: {n_inf} / {logvar_raw.numel()} "
          f"entries are -inf")


class DistributionalInputStandardizer:
    r"""
    Standardizes Gaussian-distributional inputs before they're concatenated into
    the `[mean | log-variance]` layout expected by GaussianExpectedPolynomialKernel
    (and GaussianSymmetrizedKLKernel).

    The two channels are standardized differently, on purpose:

    - **Mean channel**: ordinary per-dimension z-scoring,
      :math:`(\mu - \text{loc}) / \text{scale}`, fit on the training data.
    - **Variance channel**: per-dimension *scaling only* (no shift),
      :math:`\sigma^2 / \text{scale}`. Variance is never mean-centered, because
      (a) it must stay non-negative, and (b) leaving :math:`\sigma^2 = 0` mapped
      to exactly :math:`0` preserves the property that
      GaussianExpectedPolynomialKernel collapses exactly to the ordinary
      PolynomialKernel on point-mass (zero-variance) inputs. Centering the
      variance channel would break both of those.

    Fit statistics on the training set only, then reuse them (via `transform`) on
    validation/test data. Fitting separately on test data would let the kernel
    compare train and test points using different scales for the same physical
    quantity, silently corrupting the kernel matrix.

    Example:
        >>> standardizer = DistributionalInputStandardizer().fit(train_mean, train_var)
        >>> train_x = standardizer.transform(train_mean, train_var)
        >>> test_x = standardizer.transform(test_mean, test_var)  # reuses train stats
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_loc: torch.Tensor | None = None
        self.mean_scale: torch.Tensor | None = None
        self.var_scale: torch.Tensor | None = None

    def fit(self, mean: torch.Tensor, var: torch.Tensor, var_floor_value: float | None = None,
            min_informative: int = 3) -> "DistributionalInputStandardizer":
        """Compute standardization statistics from training means/variances (shape N x d each).

        var_floor_value: the floor value VarianceFloor clamped `var` to (its fitted
            `eps_`), if you're standardizing floored variance. Pass this so the scale
            statistic is computed only from genuinely-informative (non-floored) entries.
            Without it, a dimension where most molecules were floored to an identical
            value (e.g. "fully rigid" molecules with sigma==0, per VarianceFloor's own
            docstring) can make std() collapse toward zero -- dividing by that near-zero
            scale is what produces the catastrophic blow-up (var-term ~1e17+) rather than
            a well-conditioned kernel input.
        min_informative: a dimension needs at least this many non-floored values to trust
            its own std. Dimensions with fewer borrow the MEDIAN scale of the
            well-informed dimensions instead of an arbitrary tiny constant -- this keeps
            near-fully-rigid dimensions from re-introducing the same blow-up.
        """
        self.mean_loc = mean.mean(dim=0, keepdim=True)
        self.mean_scale = mean.std(dim=0, keepdim=True)

        def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
            nonzero = scale[scale > 0]
            rel_floor = nonzero.min() * 1e-6 if nonzero.numel() > 0 else torch.tensor(self.eps)
            return torch.where(scale > 0, scale, rel_floor.to(scale))

        self.mean_scale = _safe_scale(self.mean_scale)

        if var_floor_value is None:
            self.var_scale = _safe_scale(var.std(dim=0, keepdim=True))
        else:
            d = var.shape[-1]
            per_dim_scale = torch.full((1, d), float("nan"), dtype=var.dtype, device=var.device)
            for j in range(d):
                informative = var[:, j][var[:, j] > var_floor_value]
                if informative.numel() >= min_informative:
                    per_dim_scale[0, j] = informative.std()
            if torch.isnan(per_dim_scale).all():
                # no dimension has enough informative signal -- fall back to the
                # relative-floor guard as a last resort, and let the caller know.
                print("[warn] DistributionalInputStandardizer: no dimension had "
                      f">= {min_informative} values above var_floor_value; falling back "
                      "to a relative-scale guard. Check that VarianceFloor's floor "
                      "isn't clamping essentially all training variance to one value.")
                per_dim_scale = _safe_scale(var.std(dim=0, keepdim=True))
            else:
                fallback = per_dim_scale[~torch.isnan(per_dim_scale)].median()
                n_borrowed = torch.isnan(per_dim_scale).sum().item()
                if n_borrowed > 0:
                    print(f"[info] DistributionalInputStandardizer: {n_borrowed}/{d} "
                          f"dimensions had < {min_informative} informative (non-floored) "
                          "values; using the median scale of the other dimensions for them.")
                per_dim_scale = torch.where(torch.isnan(per_dim_scale), fallback, per_dim_scale)
            self.var_scale = _safe_scale(per_dim_scale)
        return self

    def transform(self, mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply previously-fit statistics, returning a ready-to-use [mean | log-variance] tensor."""
        if self.mean_loc is None:
            raise RuntimeError("Call .fit(train_mean, train_var) before .transform(...)")
        mean_std = (mean - self.mean_loc) / self.mean_scale
        var_scaled = (var / self.var_scale).clamp_min(self.eps)

        return mean_std, var_scaled

    def fit_transform(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Fit on this data, then transform it. Only ever call this on training data."""
        return self.fit(mean, var).transform(mean, var)