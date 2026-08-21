
from omegaconf import DictConfig, OmegaConf
from .gpr_uncertain import (
    load_descriptors_npz,
    load_freesolv_labels,
    kfold_cv_mae_gpr_deterministic,
    split_train_test_by_fraction
)
from .compute_soap_descriptors import compute_soap_descriptors


def objective_function(cfg: DictConfig):

    generate_on_fly = OmegaConf.select(cfg, "data.generate_on_fly", default=False)

    if generate_on_fly:
        ids, mu, sigma, _ = compute_soap_descriptors(cfg)
        ids = ids.astype(int)
    else:
        data = load_descriptors_npz(cfg.data.npz_path)
        mu = data["mu"]
        sigma = data["sigma"]
        ids = data["ids"].astype(int)

    y_all = load_freesolv_labels(cfg.data.label_path)
    y = y_all[ids]

    tr_idx, te_idx = split_train_test_by_fraction(len(ids), cfg.data.train_frac, seed=0)
    mu_tr, sg_tr, y_tr = mu[tr_idx], sigma[tr_idx], y[tr_idx]

    mean_cv_mae = kfold_cv_mae_gpr_deterministic(mu_tr, y_tr, cfg)

    return mean_cv_mae
