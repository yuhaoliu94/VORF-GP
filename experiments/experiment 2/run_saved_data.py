from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVR

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_data import (
    SPLIT_STATES, add_tabular_args, load_tabular, new_output,
    normalize, save_tables, split_record, write_json,
)

package = types.ModuleType("benchmark")
package.__path__ = [str(SCRIPT_DIR.parents[1] / "benchmark")]
sys.modules["benchmark"] = package
features = importlib.import_module("benchmark.orthogonal_random_feature")
TRANSFORMERS = {"orf": features.OrthogonalRandomFeature, "sorf": features.StructuredOrthogonalRandomFeature}


def load_gp_runner():
    path = SCRIPT_DIR.parent / "experiment 3" / "run_saved_data.py"
    spec = importlib.util.spec_from_file_location("saved_tabular_gp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_hard(data, dataset: str, copy: int, method: str, quick: bool) -> dict:
    seed = SPLIT_STATES[dataset][copy - 1]
    if quick:
        data = load_gp_runner().small_split(data)
    normalized = normalize(data)
    started = time.perf_counter()
    transformer = TRANSFORMERS[method](n_components=100, gamma=0.1, use_offset=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        x_train = transformer.fit_transform(normalized["x_train"])
        x_test = transformer.transform(normalized["x_test"])
    model = LinearSVR(dual=False, C=1, loss="squared_epsilon_insensitive",
                      random_state=seed, max_iter=5000, tol=1e-4)
    model.fit(x_train, normalized["y_train"].ravel())
    predictions = model.predict(x_test)[:, None]
    error = (predictions - normalized["y_test"]) * normalized["y_std"]
    return {
        "dataset": dataset, "copy": copy, "method": method, "quick": quick,
        "rmse_std": float(np.sqrt(np.mean((error / normalized["y_test_std"]) ** 2))),
        "rmse_orig": float(np.sqrt(np.mean(error ** 2))),
        "split_state": seed, "feature_seed": seed, "gamma": 0.1,
        "requested_features": 100, "actual_features": transformer.n_components,
        "train_rows": len(data.x_train), "test_rows": len(data.x_test),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 2: VORF/ORF/SORF on saved CSV splits.")
    add_tabular_args(parser)
    parser.add_argument("--methods", nargs="+", choices=("vorf", "orf", "sorf"), default=["vorf", "orf", "sorf"])
    args = parser.parse_args()
    loaded = load_tabular(args)
    if args.check_data:
        return
    gp = load_gp_runner()
    output = new_output(SCRIPT_DIR, args.results_dir)
    write_json(output / "config.json", {
        "experiment": 2, "args": vars(args), "split_states": SPLIT_STATES,
        "gamma": 0.1, "features": 100, "svr_max_iter": 5000, "svr_tol": 1e-4,
        "vorf_settings": gp.SETTINGS,
        "data": [split_record(d) for d in loaded.values()],
        "historical_difference": "Historical ORF/SORF used splits 0..4; this run uses saved CSV membership",
    })
    rows = []
    for (dataset, copy), data in loaded.items():
        for method in dict.fromkeys(args.methods):
            print(f"Running {method}/{dataset}_{copy}", flush=True)
            row = (gp.run_gp(data, dataset, copy, "vorf", args.quick) if method == "vorf"
                   else run_hard(data, dataset, copy, method, args.quick))
            rows.append(row)
            save_tables(output, rows, (2,))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
