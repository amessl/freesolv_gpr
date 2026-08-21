import torch
import numpy as np
import sys
from hydra import initialize, compose
from omegaconf import DictConfig
from src.bayes_opt import BayesianOptimizer

def run_bayes_opt(config: DictConfig) -> None:

    optimizer = BayesianOptimizer(config)

    y_init, X_init = optimizer.get_initial_samples()

    # y_init is 2D array (n_samples, n_objectives)
    y_init = np.asarray(y_init)
    if y_init.ndim == 1:
        y_init = y_init[:, None] #

    optimizer.X_train = torch.as_tensor(X_init, dtype=torch.double)
    optimizer.y_train = torch.as_tensor(y_init, dtype=torch.double)

    optimizer.fit_surrogate()

    x_list, y_list = [], []
    prev_mse = None

    for counter in range(1, 1 + config.bayes_opt.runs):
        print(f'Iteration {counter}: \n'
              '-'*12)
        x, y = optimizer.update_surrogate()


        if isinstance(y, np.ndarray) and y.ndim > 1:  # Multi-objective case
            y = y.tolist()
        else:
            y = float(y)  # For single-objective

        x_list.append(x)
        y_list.append(y)

        if prev_mse is not None and np.abs(prev_mse - np.mean(y)) < config.bayes_opt.tolerance:
            print(f"Optimization finished after {counter} iterations. MAE={y}, Hyperparams: {x}")
            break

        prev_mse = np.mean(y)

    else:
        print(f"No convergence after {counter} iterations. Last MAE={y}, Hyperparams: {x}")



if __name__ == "__main__":

    overrides = sys.argv[1:]

    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)

    run_bayes_opt(config=cfg)