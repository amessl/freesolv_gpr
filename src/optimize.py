import botorch.exceptions.errors
import torch
import numpy as np
import sys
from hydra import initialize, compose
from omegaconf import DictConfig
from src.bayes_opt import BayesianOptimizer
import warnings

warnings.filterwarnings("ignore")

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

    prev_mse = None

    for counter in range(1, 1 + config.bayes_opt.runs):
        print(f'Iteration {counter}:')
        print('-' * 36)

        x, y = optimizer.update_surrogate()
        current_mse = np.mean(y)

        print(f"Candidate error: {y}")
        print('-'*36)

        if prev_mse is not None:
            mse_change = np.abs(prev_mse - current_mse)

            if np.isclose(prev_mse, current_mse):
                print("MSE unchanged — continuing optimization.")

            elif mse_change < config.bayes_opt.tolerance:
                print(
                    f"Optimization finished after {counter} iterations. "
                    f"MAE={y}, Hyperparams: {x}"
                )
                break

        prev_mse = current_mse

    else:
        print(
            f"No convergence after {counter} iterations. "
            f"Last MAE={y}, Hyperparams: {x}"
        )



if __name__ == "__main__":

    overrides = sys.argv[1:]

    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)

    run_bayes_opt(config=cfg)