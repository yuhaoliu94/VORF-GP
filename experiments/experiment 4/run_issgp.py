from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from common import (
    DEFAULT_NPZ_PATH,
    DEFAULT_RESULTS_DIR,
    base_result_payload,
    iter_minibatches,
    make_splits,
    median_lengthscale,
    regression_metrics,
    timestamp_token,
    write_json,
)


class ISSGPRegressor:

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        lengthscale: float,
        noise_variance: float,
        kernel_variance: float,
        prior_variance: float,
        seed: int,
    ) -> None:
        self.input_dim = input_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance
        self.kernel_variance = kernel_variance
        self.prior_variance = prior_variance
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=1.0 / lengthscale, size=(input_dim, n_features))
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        self.precision = np.eye(n_features, dtype=np.float64) / prior_variance
        self.rhs = np.zeros((n_features, 1), dtype=np.float64)
        self._factor: tuple[np.ndarray, bool] | None = None
        self._weights: np.ndarray | None = None

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 * self.kernel_variance / self.n_features) * np.cos(X @ self.W + self.phase)

    def update_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        Phi = self.features(X)
        self.precision += (Phi.T @ Phi) / self.noise_variance
        self.rhs += (Phi.T @ y.reshape(-1, 1)) / self.noise_variance
        self._factor = None
        self._weights = None

    def finalize(self) -> None:
        self._factor = cho_factor(self.precision, lower=True, check_finite=False)
        self._weights = cho_solve(self._factor, self.rhs, check_finite=False)

    def predict(self, X: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if self._factor is None or self._weights is None:
            self.finalize()

        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        assert self._factor is not None
        assert self._weights is not None
        for start in range(0, X.shape[0], batch_size):
            Phi = self.features(X[start : start + batch_size])
            mean = Phi @ self._weights
            solved = cho_solve(self._factor, Phi.T, check_finite=False)
            latent_var = np.einsum("ij,ji->i", Phi, solved)
            obs_var = self.noise_variance + np.maximum(latent_var, 0.0)
            means.append(mean)
            variances.append(obs_var[:, None])
        return np.vstack(means), np.vstack(variances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ISSGP baseline on prepared Airline data.")
    parser.add_argument("--data-npz", "--data-dir", dest="data_npz", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=37)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--lengthscale", default="auto", help="Positive scalar or 'auto' for median heuristic.")
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--shuffle-train", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_lengthscale(value: str, X_train: np.ndarray, seed: int) -> float:
    if value == "auto":
        return median_lengthscale(X_train, seed=seed)
    lengthscale = float(value)
    if lengthscale <= 0.0:
        raise ValueError("--lengthscale must be positive")
    return lengthscale


def main() -> None:
    args = parse_args()
    split_data = make_splits(
        data_npz=args.data_npz,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split=args.split,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    lengthscale = resolve_lengthscale(args.lengthscale, split_data.X_train, args.model_seed)
    model = ISSGPRegressor(
        input_dim=split_data.X_train.shape[1],
        n_features=args.features,
        lengthscale=lengthscale,
        noise_variance=args.noise_std**2,
        kernel_variance=args.kernel_variance,
        prior_variance=args.prior_variance,
        seed=args.model_seed,
    )

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for pass_index in range(args.passes):
        print(f"ISSGP pass {pass_index + 1}/{args.passes}")
        for X_batch, y_batch in iter_minibatches(
            split_data.X_train,
            split_data.y_train,
            batch_size=args.batch_size,
            shuffle=args.shuffle_train,
            rng=rng,
        ):
            model.update_batch(X_batch, y_batch)
    model.finalize()
    train_seconds = time.perf_counter() - started

    predict_started = time.perf_counter()
    pred_mean, pred_var = model.predict(split_data.X_test, batch_size=args.predict_batch_size)
    predict_seconds = time.perf_counter() - predict_started
    metrics = regression_metrics(
        split_data.y_test,
        pred_mean,
        pred_var,
        y_mean=split_data.y_mean,
        y_std=split_data.y_std,
    )

    payload = base_result_payload("issgp", args, split_data)
    payload.update(
        {
            "hyperparameters": {
                "features": args.features,
                "lengthscale": lengthscale,
                "noise_variance": args.noise_std**2,
                "kernel_variance": args.kernel_variance,
                "prior_variance": args.prior_variance,
                "passes": args.passes,
            },
            "timing": {
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": train_seconds + predict_seconds,
            },
            "metrics": metrics,
        }
    )

    output = args.results_dir / f"issgp_{timestamp_token()}.json"
    write_json(output, payload)
    print(f"Wrote {output}")
    print(
        "ISSGP metrics: "
        f"rmse_orig={metrics['rmse_orig']:.4f}, "
        f"rmse_std={metrics['rmse_std']:.4f}, "
        f"nll_orig={metrics['nll_orig']:.4f}"
    )


if __name__ == "__main__":
    main()
