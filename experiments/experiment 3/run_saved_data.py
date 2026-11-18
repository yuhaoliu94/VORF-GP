from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "src"))

import numpy as np

from saved_data import (
    SPLIT_STATES, SavedSplit, add_tabular_args, load_tabular, new_output,
    save_tables, split_record, training_stream, write_json,
)
from orf_gp.config import GPConfig
from orf_gp.models import OnlineGP
from orf_gp.monitor import OnlineMonitorValidate
from orf_gp.utils import set_global_seeds

# settings for each dataset, including model seed, number of repeats, MC test samples, alpha, and learning rates for different parameters. These are the best settings found in prior grid search experiments, where the model_seed is fixed and not optimized.
SETTINGS = {
    "concrete": {"model_seed": 37, "repeats": 100, "mc_test": 1000, "alpha": 0.005,
                 "lr": {"mu": 0.01, "b": 0.0001, "l": 0.1, "sigma": 0.1}},
    "kin8nm": {"model_seed": 37, "repeats": 200, "mc_test": 500, "alpha": 0.01,
               "lr": {"mu": 0.01, "b": 0.00003, "l": 1.0, "sigma": 1.0}},
    "powerplant": {"model_seed": 37, "repeats": 200, "mc_test": 500, "alpha": 0.01,
                   "lr": {"mu": 0.04, "b": 0.0006, "l": 0.0003, "sigma": 0.7}},
    "protein": {"model_seed": 1, "repeats": 150, "mc_test": 1000, "alpha": 0.001,
                "lr": {"mu": 0.1, "b": 0.01, "l": 0.1, "sigma": 0.1}},
}
KERNELS = {"vorf": "Orthogonal", "rbf": "RBF", "maxwell": "Maxwell"}


class SilentMonitor:
    def check(self, model, losses):
        pass


def small_split(data: SavedSplit) -> SavedSplit:
    return SavedSplit(data.x_train[:64], data.y_train[:64], data.x_test[:32],
                      data.y_test[:32], data.feature_names, data.directory)


def run_gp(data: SavedSplit, dataset: str, copy: int, method: str, quick: bool = False) -> dict:
    settings = SETTINGS[dataset]
    seed = SPLIT_STATES[dataset][copy - 1]
    protein_vorf = dataset == "protein" and method == "vorf"
    power_vorf = dataset == "powerplant" and method == "vorf"
    validation = 0.2 if dataset == "powerplant" or protein_vorf else 0.0
    if quick:
        data = small_split(data)
    repeats = 1 if quick else settings["repeats"]
    mc_test = 3 if quick else settings["mc_test"]
    with contextlib.redirect_stdout(io.StringIO()):
        set_global_seeds(seed)
    stream = training_stream(data, seed, repeats, validation, 1e-8 if protein_vorf else 0.0)
    x_train, y_train = stream["x_train"], stream["y_train"]
    x_val, y_val = stream["x_val"], stream["y_val"]
    steps = len(x_train)
    interval = max(1, steps // 10)
    features = 16 if quick else 100
    config = GPConfig(
        rff_model=KERNELS[method], model_seed=settings["model_seed"], J=features,
        Dx=x_train.shape[1], Dy=y_train.shape[1], sigma_l=1.0, gamma_obs=1.0, MC=1,
        beta_kl=1e-4 / x_train.shape[1], beta_ortho=1e-3 / features,
        meta_lr=dict(settings["lr"]), alpha_meta=settings["alpha"], fixed_step=-1,
        log_interval=10000 if protein_vorf else interval, verbal=False,
    )
    monitor = (OnlineMonitorValidate(x_val, y_val, log_interval=10000, verbal=False)
               if protein_vorf else SilentMonitor())
    model = OnlineGP(config=config, monitor=monitor)
    model.initialize_structure()
    best_model = None
    best_rmse = best_nll = math.inf
    best_step = 0
    checks = 0
    mc_val = 3 if quick else 100

    def check_validation(step: int) -> None:
        nonlocal best_model, best_rmse, best_nll, best_step, checks
        candidate = deepcopy(model) if power_vorf else model
        _, _, nll, _, rmse = candidate.observation_layer.predict_batch_mc(x_val, y_val, MC=mc_val)
        checks += 1
        better = rmse < best_rmse
        if power_vorf:
            better = better or (math.isclose(rmse, best_rmse) and nll < best_nll)
        if better:
            best_rmse, best_nll, best_step = float(rmse), float(nll), step
            best_model = deepcopy(model)

    started = time.perf_counter()
    if power_vorf:
        check_validation(0)
    for t, (x, y) in enumerate(zip(x_train, y_train), start=1):
        losses, _, _ = model.step(y, x, MC=1)
        if not np.isfinite(np.asarray(losses)).all():
            raise FloatingPointError(f"{dataset}_{copy}/{method}: non-finite loss at step {t}")
        if dataset == "powerplant":
            if power_vorf and t % interval == 0:
                check_validation(t)
            elif not power_vorf and (t % interval == 0 or t == steps):
                check_validation(t)
        if t % interval == 0:
            print(f"  {method}/{dataset}_{copy}: {t}/{steps}", flush=True)
    if power_vorf and best_step != steps:
        check_validation(steps)
    selected = best_model if dataset == "powerplant" and best_model is not None else model
    predictions, _, nll, _, _ = selected.observation_layer.predict_batch_mc(
        stream["x_test"], stream["y_test"], MC=mc_test,
    )
    error = (predictions - stream["y_test"]) * stream["y_std"]
    metrics = {
        "rmse_std": float(np.sqrt(np.mean((error / stream["y_test_std"]) ** 2))),
        "rmse_orig": float(np.sqrt(np.mean(error ** 2))),
        "nll_orig": float(nll + np.log(stream["y_std"]).sum()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"{dataset}_{copy}/{method}: non-finite test metric")
    return {
        "dataset": dataset, "copy": copy, "method": method, "quick": quick,
        **metrics, "split_state": seed, "model_seed": settings["model_seed"],
        "features": features, "repeats": repeats, "mc_test": mc_test,
        "validation_fraction": validation, "validation_checks": checks,
        "selected_step": best_step if dataset == "powerplant" else steps,
        "train_rows": stream["train_base_size"], "train_steps": steps,
        "test_rows": len(stream["x_test"]), "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tables 3/4: train GP models on saved CSV splits.")
    add_tabular_args(parser)
    parser.add_argument("--methods", nargs="+", choices=tuple(KERNELS), default=list(KERNELS))
    args = parser.parse_args()
    loaded = load_tabular(args)
    if args.check_data:
        return
    output = new_output(SCRIPT_DIR, args.results_dir)
    write_json(output / "config.json", {
        "experiment": 3, "args": vars(args), "settings": SETTINGS,
        "split_states": SPLIT_STATES, "data": [split_record(d) for d in loaded.values()],
        "literature_rows": "BSGP, IPVI GP, SGHMC GP, SVGP are external; not rerun here",
    })
    rows = []
    for (dataset, copy), data in loaded.items():
        for method in dict.fromkeys(args.methods):
            print(f"Running {method}/{dataset}_{copy}", flush=True)
            rows.append(run_gp(data, dataset, copy, method, args.quick))
            save_tables(output, rows, (3, 4))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
