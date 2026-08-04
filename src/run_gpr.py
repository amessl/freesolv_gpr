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

from .gpr_uncertain import (
    TrainConfig,
    evaluate_gpr_uncertain,
    evaluate_gpr_deterministic,
    load_descriptors_npz,
    load_freesolv_labels,
    train_gpr_uncertain,
    train_gpr_deterministic,
    kfold_cv_mae_gpr_uncertain,
    kfold_cv_mae_gpr_deterministic
)


def main(cfg: DictConfig):
    data = load_descriptors_npz(cfg.data.npz_path)
    mu = data["mu"]
    sigma = data["sigma"]
    ids = data["ids"].astype(int)

    y_all = load_freesolv_labels(cfg.data.label_path)
    y = y_all[ids]

    # Train/test split
    from .gpr_uncertain import split_train_test_by_fraction

    tr_idx, te_idx = split_train_test_by_fraction(len(ids), cfg.data.train_frac, seed=0)
    mu_tr, sg_tr, y_tr = mu[tr_idx], sigma[tr_idx], y[tr_idx]
    mu_te, sg_te, y_te = mu[te_idx], sigma[te_idx], y[te_idx]

    if cfg.training.mode == "uncertain":
        model, likelihood = train_gpr_uncertain(mu_tr, sg_tr, y_tr, cfg)
        metrics = evaluate_gpr_uncertain(model, likelihood, mu_te, sg_te, y_te)
        if cfg.training.cross_val:
            mean_cv_mae = kfold_cv_mae_gpr_uncertain(mu_tr, sg_tr, y_tr, cfg)
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
