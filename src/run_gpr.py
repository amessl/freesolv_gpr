"""
Command-line utility to train/test a GPyTorch GPR model with input uncertainty
using descriptor means/stds stored in an .npz file.

Example:
    python -m src.run_gpr \
        --npz data/reps/total/descriptors_tier1.npz \
        --labels data/freesolv.csv \
        --epochs 150 --mc-samples-train 1 --mc-samples-test 32
"""

from hydra import initialize, compose
from omegaconf import DictConfig
import torch

from .gpr_uncertain import (
    TrainConfig,
    evaluate_gpr_uncertain,
    evaluate_gpr_deterministic,
    load_descriptors_npz,
    load_freesolv_labels,
    train_gpr_uncertain,
    train_gpr_deterministic,
    kfold_cv_mae_gpr_uncertain,
    kfold_cv_mae_gpr_deterministic,
    get_predictive_uncertainty_uncertain
)
from src.preprocess import VarianceFloor, DistributionalInputStandardizer
from src.preprocess import diagnose_uncertain_input

def main(cfg: DictConfig):
    data = load_descriptors_npz(cfg.data.npz_path)
    mu = data["mu"]
    sigma = data["sigma"]
    var = sigma**2
    ids = data["ids"].astype(int)

    print("Running....")

    y_all = load_freesolv_labels(cfg.data.label_path)
    y = y_all[ids]

    # Train/test split
    from .gpr_uncertain import split_train_test_by_fraction

    tr_idx, te_idx = split_train_test_by_fraction(len(ids), cfg.data.train_frac, seed=0)
    mu_tr, var_tr, y_tr = mu[tr_idx], var[tr_idx], y[tr_idx]
    mu_te, var_te, y_te = mu[te_idx], var[te_idx], y[te_idx]


    percentile = cfg.training.variance_floor_percentile
    floor = VarianceFloor(percentile=percentile)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32



    if cfg.training.mode == "uncertain":

        # Diagnose uncertain inputs before preprocessing
        print('Diagnose uncertain inputs:')
        diagnose_uncertain_input(torch.as_tensor(mu_tr, dtype=dtype, device=device),
                                 torch.as_tensor(var_tr, dtype=dtype, device=device))

        mu_tr = torch.as_tensor(mu_tr, dtype=dtype, device=device)
        mu_te = torch.as_tensor(mu_te, dtype=dtype, device=device)

        var_tr = floor.fit_transform(torch.as_tensor(var_tr, dtype=dtype, device=device))
        var_te = floor.transform(torch.as_tensor(var_te, dtype=dtype, device=device))

        # Rescale mu/var to a comparable magnitude before they reach the kernel. Without
        # this, var (fit from genuinely tiny raw Coulomb-descriptor variances) can sit
        # 10+ orders of magnitude below mu's scale, which both risks Cholesky/PSD failures
        # at kernel_degree >= 2 and makes the kernel's uncertainty term negligible relative
        # to its mean term (silently degrading to a near-deterministic model).
        standardizer = DistributionalInputStandardizer().fit(mu_tr, var_tr, var_floor_value=floor.eps_)
        mu_tr, var_tr = standardizer.transform(mu_tr, var_tr)
        mu_te, var_te = standardizer.transform(mu_te, var_te)

        model, likelihood = train_gpr_uncertain(mu_tr, var_tr, y_tr, cfg)
        metrics = evaluate_gpr_uncertain(model, likelihood, mu_te, var_te, y_te)


        if cfg.training.cross_val:
            mean_cv_mae = kfold_cv_mae_gpr_uncertain(mu_tr, var_tr, y_tr, cfg)
            print(f'CV-MAE:{mean_cv_mae}')
    else:
        model, likelihood = train_gpr_deterministic(mu_tr, y_tr, cfg)
        metrics = evaluate_gpr_deterministic(model, likelihood, mu_te, y_te)
        if cfg.training.cross_val:
            mean_cv_mae = kfold_cv_mae_gpr_deterministic(mu_tr, y_tr, cfg)
            print(f'CV-MAE:{mean_cv_mae}')

    print({k: float(v) for k, v in metrics.items()})



if __name__ == "__main__":
    with initialize(config_path="../conf", version_base="1.1"):
        cfg=compose(config_name="config")

    main(cfg)