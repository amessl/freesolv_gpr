
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from omegaconf import OmegaConf, DictConfig
from typing import Union, Tuple, List
from src.sampler import SamplingUtils
from src.objective import objective_function


class BayesianOptimizer:

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.sampler = SamplingUtils(config, config.bayes_opt.n_init_samples)
        self.X_train = None
        self.y_train = None

    def get_initial_samples(self) -> Tuple[List, List]:
        initial_samples = self.sampler.eval_lhs_grid()

        return initial_samples

    def assert_dtypes(self, input_list: Union[List[float], torch.Tensor]) -> Tuple[torch.Tensor, tuple]:
        dtypes = self.config.bayes_opt.dtype_pattern
        orig_shape = (1, len(dtypes))

        output_list = []

        if isinstance(input_list, torch.Tensor):
            input_list = input_list.flatten().tolist()

        for dtype, val in zip(dtypes, input_list):
            if dtype == 'float':
                val = float(val)
            elif dtype == 'int':
                val = int(np.round(val))

            output_list.append(val)

        output_tensor = torch.tensor(output_list)

        return output_tensor, orig_shape

    def fit_surrogate(self) -> Tuple[torch.Tensor, SingleTaskGP]:

        X_train = torch.tensor(self.X_train).to(torch.double)
        # y_train is already shaped (n, 1); avoid adding an extra dimension
        y_train = torch.tensor(self.y_train).to(torch.double)

        surrogate = SingleTaskGP(  # model (Gaussian Process with default RBF kernel)
                    train_X=X_train, # X of init_samples
                    train_Y=y_train, # y of init_samples
                    input_transform=Normalize(d=self.X_train.size()[1]),
                    outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
        fit_gpytorch_mll(mll)

        logEI = LogExpectedImprovement(model=surrogate, best_f=y_train.min(), maximize=False) # watch out for sign of objective

        l_bounds, u_bounds = self.sampler.get_bounds()
        bounds = torch.stack([torch.tensor(l_bounds), torch.tensor(u_bounds)]).to(torch.double)

        candidate, acq_value = optimize_acqf(acq_function=logEI, bounds=bounds,
                                             q=1, num_restarts=self.config.bayes_opt.num_restarts,
                                             raw_samples=self.config.bayes_opt.raw_samples)
        # 1. Draw raw_samples candidates uniformly from the domain.
        # 2. Evaluate the acquisition function on them.
        # 3. Pick the best num_restarts initial points.
        # 4. Start gradient - based optimizations from each of those and pick the best result overall.

        # q=1: single-point Bayesian optimization(sequential).

        # q>1: batch optimization — find multiple candidates jointly.
        # If only n random initial points are generated, more than n optimization runs cannot be started
        # There are not enough distinct starting positions
        candidate, orig_shape = self.assert_dtypes(candidate.flatten().tolist())

        return candidate.view(orig_shape), surrogate


    def update_surrogate(self) -> Tuple[List, torch.Tensor]:

        candidate, surrogate = self.fit_surrogate()

        while True:
            candidate, orig_shape = self.assert_dtypes(candidate)

            print(f"New candidate: {candidate}")

            candidate_list = candidate.flatten().tolist()

            default_hyperparams = OmegaConf.to_container(self.config.reps.soap_params, resolve=True)

            for key, param in zip(default_hyperparams.keys(), candidate_list):
                default_hyperparams[key] = param

            # Clone cfg
            updated_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))

            # Update representation-specific hyperparams
            for key in default_hyperparams.keys():
                updated_cfg.reps.soap_params[key] = default_hyperparams[key]

            #print(f"Updated config: {updated_cfg}")

            err_value = objective_function(updated_cfg)

            if not np.isnan(err_value):
                break

            print(f"Candidate {candidate_list} produced NaN. Rejecting and proposing new candidate.")
            # Add failed candidate to training data with a very bad value to avoid it in the next iteration
            # Use a value that is worse than the current worst but not so large it ruins standardization
            current_max = self.y_train[self.y_train < 1e8].max() if (self.y_train < 1e8).any() else torch.tensor(1.0)
            failed_err = torch.tensor([[current_max + 1.0]], dtype=self.y_train.dtype)
            self.X_train = torch.cat([self.X_train, candidate.view(orig_shape)], dim=0)
            self.y_train = torch.cat([self.y_train, failed_err], dim=0)
            
            candidate, surrogate = self.fit_surrogate()

        # Ensure err has shape (1, 1) to match y_train's (n, 1)
        err = torch.as_tensor(err_value, dtype=self.y_train.dtype).view(1, 1)

        self.X_train = torch.cat([self.X_train, candidate.view(orig_shape)], dim=0)
        self.y_train = torch.cat([self.y_train, err], dim=0)

        return candidate_list, err_value
