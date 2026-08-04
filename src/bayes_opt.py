
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
from src.objective import ObjectiveFunction


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

    def fit_surrogate(self) -> torch.Tensor:

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

        logEI = LogExpectedImprovement(model=surrogate, best_f=y_train.max()) # watch out for sign of objective

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

        return candidate.view(orig_shape)

    def update_surrogate(self) -> Tuple[List, torch.Tensor]:

        candidate = self.fit_surrogate()
        candidate, orig_shape = self.assert_dtypes(candidate)

        print(f"New candidate: {candidate}")

        candidate_list = candidate.flatten().tolist()

        default_hyperparams = OmegaConf.to_container(self.config.rep.soap_params, resolve=True)

        for key, param in zip(default_hyperparams.keys(), candidate_list):
            default_hyperparams[key] = param

        # Clone cfg
        updated_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))

        # Update representation-specific hyperparams
        for key in default_hyperparams.keys():
            updated_cfg.rep.soap_params[key] = default_hyperparams[key]

        print(f"Updated config: {updated_cfg}")

        obj_func = ObjectiveFunction(config=updated_cfg)
        err_value = 0

        if self.config.bayes_opt.learn_type_bo == 'ST':
            err_value = -obj_func.objective_ST()[1]
        elif self.config.bayes_opt.learn_type_bo == 'MT':
            err_value = -obj_func.objective_MT()[1]

        # Ensure err has shape (1, 1) to match y_train's (n, 1)
        err = torch.as_tensor(err_value, dtype=self.y_train.dtype).view(1, 1)

        self.X_train = torch.cat([self.X_train, candidate.view(orig_shape)], dim=0)
        self.y_train = torch.cat([self.y_train, err], dim=0)

        return candidate_list, err_value



# TODO: cleanup and docs


# Meeting 05.03.26
# TODO: Use Train/validation split of dyes1-3 to optimize but also test on this dataset before testing transfer to dye4 (splitting has to be modified for that)
# TODO: Use Bayesian Optimization instead of Pareto since it is intended for multi-objective where objectives can be conflicting
# TODO: Optimize on excitation energies first and then on oscillator strengths (maybe use mean MSE as objective)
# TODO: Check scaling of different properties

# TODO: Create separate method for splitting specifically dyes1,2,3 into train and validation and test set and dyes4 as external validation (validate splitting method by comparing results with original method)

# Meeting 12.03.26
# TODO: MSE value of 0.1 eV (chemical accuracy 0.01-0.05 eV), keep in mind thst MSE becomes smaller
# TODO: Find literature for testing XC-functionals for excitation energies (test sets for excitation energies)
# TODO: Average of excitation MSEs as objective function