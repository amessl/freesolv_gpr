import torch
import numpy as np
from omegaconf import OmegaConf
from src.bayes_opt import BayesianOptimizer
import src.objective

# Mock objective function to simulate NaN for specific candidates
original_objective = src.objective.objective_function

def mock_objective(cfg):
    # Simulate NaN if rcut is between 3.0 and 4.0
    rcut = cfg.reps.soap_params.rcut
    if 3.0 <= rcut <= 4.0:
        print(f"DEBUG: Mocking NaN for rcut={rcut}")
        return np.nan
    # Instead of calling original_objective, return a dummy value to avoid needing real data
    return -0.42

src.objective.objective_function = mock_objective

def test_nan_rejection():
    # Load a minimal config
    config = OmegaConf.create({
        "bayes_opt": {
            "n_init_samples": 2,
            "num_restarts": 2,
            "raw_samples": 10,
            "dtype_pattern": ["float", "int", "int", "float"],
            "grid": {
                "rcut": [2.0, 6.0],
                "nmax": [1, 7],
                "lmax": [1, 7],
                "sigma": [0.1, 1.5]
            },
            "runs": 5,
            "tolerance": 0.001
        },
        "reps": {
            "soap_params": {
                "rcut": 3.0,
                "nmax": 4,
                "lmax": 4,
                "sigma": 0.5
            }
        },
        "data": {
            "generate_on_fly": False,
            "npz_path": "data/descriptors.npz", # Might need to exist or be mocked
            "label_path": "freesolv.csv",
            "train_frac": 0.8
        }
    })

    # We need some dummy data if we don't want it to fail on loading
    # For this test, we just want to see if the loop in update_surrogate works
    
    optimizer = BayesianOptimizer(config)
    optimizer.X_train = torch.tensor([[2.0, 1, 1, 0.1], [6.0, 7, 7, 1.5]], dtype=torch.double)
    optimizer.y_train = torch.tensor([[-1.0], [-0.5]], dtype=torch.double)

    print("Starting update_surrogate...")
    try:
        candidate_list, err_value = optimizer.update_surrogate()
        print(f"Success! Candidate: {candidate_list}, Error: {err_value}")
        assert not np.isnan(err_value)
    except Exception as e:
        print(f"Failed with exception: {e}")
        raise e

if __name__ == "__main__":
    test_nan_rejection()
