#!/usr/bin/env python3
"""
Distributional analogue of gpytorch.kernels.PolynomialKernel.

GaussianSymmetrizedKLKernel is built on gpytorch's DistributionalInputKernel base
class, whose forward() hardcodes an RBF-style exp(-distance / lengthscale)
structure. That base class cannot express a (inner_product + offset)^power form,
so this kernel subclasses gpytorch.kernels.Kernel directly instead, mirroring
PolynomialKernel's parameter conventions (raw_offset, power, Positive constraint)
while replacing the plain dot product with a distributional inner product.

Inputs use the same `batch x N x 2d` layout as GaussianSymmetrizedKLKernel: the
first d dims are means, the second d dims are log-variances of independent
per-dimension Gaussians.
"""
from __future__ import annotations

import torch

from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel


def _gaussian_moment_inner_product(dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
    r"""
    "Expected dot-product" style inner product between diagonal Gaussian distributions.

    Each distribution is represented by its sufficient statistics, mean and variance,
    concatenated as :math:`\phi(p) = [\mu, \sigma^2]`. The inner product
    :math:`\langle \phi(p), \phi(p') \rangle = \mu^\top \mu' + (\sigma^2)^\top(\sigma'^2)`
    reduces *exactly* to the ordinary dot product :math:`x^\top x'` when both
    distributions collapse to point masses (:math:`\sigma^2 \to 0`), which is what
    makes this a genuine distributional analogue of the linear/polynomial kernel's
    dot-product term — the variance contributes an extra "covariance alignment"
    term (a Frobenius inner product between diagonal covariances) on top of the
    ordinary mean-alignment term.

    Args:
        dist1: batch x n x 2d tensor. First d dims = means, second d dims = log-variances.
        dist2: batch x m x 2d tensor. Same layout as dist1.

    Returns:
        batch x n x m tensor of inner products.
    """
    num_dims = int(dist1.shape[-1] / 2)

    mu1 = dist1[..., :num_dims]
    var1 = dist1[..., num_dims:].exp()

    mu2 = dist2[..., :num_dims]
    var2 = dist2[..., num_dims:].exp()

    # sufficient-statistic embedding: [mean, variance]
    phi1 = torch.cat([mu1, var1], dim=-1)
    phi2 = torch.cat([mu2, var2], dim=-1)

    return torch.matmul(phi1, phi2.transpose(-2, -1))


class GaussianExpectedPolynomialKernel(Kernel):
    r"""
    Distributional analogue of :class:`gpytorch.kernels.PolynomialKernel`, for inputs
    that represent diagonal Gaussian distributions rather than point vectors.

    Inputs are assumed to be `batch x N x 2d` tensors, matching the convention used by
    :class:`gpytorch.kernels.GaussianSymmetrizedKLKernel`: the first `d` dimensions are
    the means, the second `d` dimensions are the log-variances.

    .. math::
        \begin{equation*}
            k(p, p') = \big(\langle \phi(p), \phi(p') \rangle + c\big)^{d}
        \end{equation*}

    where :math:`\phi(p) = [\mu_p, \sigma_p^2]` and :math:`c` is a learnable offset.

    Unlike GaussianSymmetrizedKLKernel, this does NOT subclass DistributionalInputKernel
    -- that base class's forward() is hardcoded to exp(-distance/lengthscale), which
    can't represent a polynomial (dot + c)^power structure. This subclasses Kernel
    directly instead, following the same parameter/constraint conventions as
    PolynomialKernel.

    Args:
        power: degree of the polynomial.
        offset_prior: prior over the offset parameter.
        offset_constraint: constraint on the offset parameter (default: Positive).
        inner_product_function: callable(dist1, dist2) -> batch x n x m tensor.
            Defaults to the mean/variance inner product above; pass your own to use
            a different sufficient-statistic embedding (e.g. full covariance matrices
            instead of diagonal, or a different moment combination).

    Example:
        >>> from gpytorch.kernels import ScaleKernel
        >>> # x is batch x N x 2d: first d dims mean, second d dims log-variance
        >>> kernel = ScaleKernel(GaussianExpectedPolynomialKernel(power=2))
        >>> covar = kernel(x)  # drop-in replacement for ScaleKernel(GaussianSymmetrizedKLKernel())

    Numerical stability note:
        (inner_product + offset)^power is exactly PSD in exact arithmetic for
        offset >= 0 and integer power (it's a Gram matrix, and elementwise powers of
        PSD matrices stay PSD by the Schur product theorem). In finite precision,
        power >= 2 squares (or higher-powers) the matrix's dynamic range, so if the
        mean and variance channels of your inputs are on very different scales, the
        largest and smallest eigenvalues can end up many orders of magnitude apart --
        the true smallest eigenvalue is a tiny positive number that rounds to
        slightly negative in float32, and Cholesky-based routines will report the
        matrix as not positive definite even though it mathematically is.

        Standardizing the mean channel alone is not enough; the variance channel
        must be brought to a comparable scale too. Use
        :class:`DistributionalInputStandardizer` (below) to do this consistently
        across train/test data before building kernel inputs. If issues persist
        after standardizing, try float64 and/or a larger
        `gpytorch.settings.cholesky_jitter`.
    """

    has_lengthscale = False

    def __init__(
        self,
        power: int,
        offset_prior: Prior | None = None,
        offset_constraint: Interval | None = None,
        inner_product_function=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Positive()

        self.register_parameter(
            name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )

        if torch.is_tensor(power):
            if power.numel() > 1:
                raise RuntimeError("Can't create a polynomial kernel with more than one power")
            power = power.item()
        self.power = power

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

        self.inner_product_function = inner_product_function or _gaussian_moment_inner_product

    @property
    def offset(self) -> torch.Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        offset = self.offset.view(*self.batch_shape, 1, 1)
        inner_product = self.inner_product_function(x1, x2)
        res = (inner_product + offset).pow(self.power)

        if not diag:
            return res
        return res.diagonal(dim1=-1, dim2=-2)
