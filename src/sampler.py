import numpy as np
from typing import List, Tuple
from scipy.stats import qmc
from omegaconf import OmegaConf, DictConfig
from multiprocessing import Pool
from src.objective import objective_function


class SamplingUtils:

    def __init__(self, config: DictConfig, n_init_samples) -> None:
        self.config = config
        self.n_init_samples = n_init_samples

    def worker(self, hyperparams_dict):
        updated_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))

        for key, value in hyperparams_dict.items():
            updated_cfg.reps.soap_params[key] = value

        obj = objective_function(updated_cfg)

        return obj



    def get_search_space(self) -> DictConfig:
        search_space = OmegaConf.to_container(self.config.bayes_opt.grid, resolve=True)

        return search_space

    def get_bounds(self) -> Tuple[List, List]:
        search_space = self.get_search_space()

        l_bounds = []  # lower bounds of each hyperparameter range
        u_bounds = []  # upper bounds of each hyperparameter range

        for hyperparam in search_space:
            hyperparam_list = search_space[hyperparam]
            l_bounds.append(min(hyperparam_list))
            u_bounds.append(max(hyperparam_list))

        return l_bounds, u_bounds


    def lhs_sample(self, n_hyperparams: int, n_samples: int) -> dict:

        search_space = self.get_search_space()

        sampler = qmc.LatinHypercube(d=n_hyperparams)  # e.g. four SOAP hyperparameters
        sample = sampler.random(n=n_samples)  # number of samples

        l_bounds, u_bounds = self.get_bounds()

        dtypes = [type(val) for val in l_bounds]
        print(dtypes)

        sample_scaled = qmc.scale(sample, l_bounds=l_bounds, u_bounds=u_bounds)
        sample_scaled_dict = dict()

        for col_index, hyperparam in zip(range(0,np.shape(sample_scaled)[1]), search_space.keys()):
            sample_scaled[:, col_index] = sample_scaled[:, col_index].astype(dtypes[col_index])
            sample_scaled_dict[hyperparam] = np.ndarray.tolist(sample_scaled[:, col_index])

        return sample_scaled_dict


    def eval_lhs_grid(self) -> Tuple[List, List]:

        n_hypers = len(self.get_search_space().keys())

        sample_scaled_dict = self.lhs_sample(n_hyperparams=n_hypers, n_samples=self.n_init_samples)
        print(f"Initial samples: {sample_scaled_dict}")

        param_grid = sample_scaled_dict
        param_names = list(sample_scaled_dict.keys())

        hyperparam_combinations = []

        grid = np.array([*param_grid.values()]).T.tolist()

        tasks = []

        for iteration, hyperparams in enumerate(grid, 1):
            hyperparams_dict = dict(zip(param_names, hyperparams))

            tasks.append(hyperparams_dict)

            print(hyperparams_dict)

            hyperparam_combinations.append(hyperparams)

        with Pool(processes=1) as pool:
            error_list = pool.map(self.worker, tasks)

        valid = [
            (err, hp)
            for err, hp in zip(error_list, hyperparam_combinations)
            if not np.isnan(err)
        ]

        error_list = [err for err, hp in valid]
        hyperparam_combinations = [hp for err, hp in valid]

        n_failed = self.n_init_samples - len(error_list)
        print(f"{n_failed} samples failed and were removed from initial samples")
        print(error_list)

        return error_list, hyperparam_combinations