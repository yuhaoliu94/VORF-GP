from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from common import (
    DEFAULT_NPZ_PATH,
    DEFAULT_RESULTS_DIR,
    base_result_payload,
    choose_inducing_points,
    iter_minibatches,
    make_splits,
    median_lengthscale,
    rbf_kernel,
    regression_metrics,
    timestamp_token,
    write_json,
)


class FixedInducingStreamingSparseGP:

    def __init__(
        self,
        inducing_points: np.ndarray,
        lengthscale: float,
        noise_variance: float,
        kernel_variance: float,
        jitter: float,
        variance_mode: str,
    ) -> None:
        self.Z = np.asarray(inducing_points, dtype=np.float64)
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance
        self.kernel_variance = kernel_variance
        self.jitter = jitter
        self.variance_mode = variance_mode

        Kzz = rbf_kernel(self.Z, self.Z, lengthscale, variance=kernel_variance)
        Kzz = Kzz + jitter * np.eye(self.Z.shape[0])
        self.Lzz = np.linalg.cholesky(Kzz)
        self.precision = np.eye(self.Z.shape[0], dtype=np.float64)
        self.rhs = np.zeros((self.Z.shape[0], 1), dtype=np.float64)
        self._factor: tuple[np.ndarray, bool] | None = None
        self._weights: np.ndarray | None = None

    def features(self, X: np.ndarray) -> np.ndarray:
        Kzx = rbf_kernel(self.Z, X, self.lengthscale, variance=self.kernel_variance)
        return solve_triangular(self.Lzz, Kzx, lower=True, check_finite=False).T

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
            posterior_latent_var = np.einsum("ij,ji->i", Phi, solved)
            q_diag = np.sum(Phi * Phi, axis=1)
            correction = 0.0
            if self.variance_mode == "fitc":
                correction = np.maximum(self.kernel_variance - q_diag, 0.0)
            obs_var = self.noise_variance + correction + np.maximum(posterior_latent_var, 0.0)
            means.append(mean)
            variances.append(obs_var[:, None])
        return np.vstack(means), np.vstack(variances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SSGP baseline on prepared Airline data.")
    parser.add_argument("--data-npz", "--data-dir", dest="data_npz", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--backend",
        choices=("numpy", "gpflow"),
        default="numpy",
        help="numpy is a fixed-inducing streaming sparse GP; gpflow uses the official thangbui repo adapter.",
    )
    parser.add_argument("--split", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=37)
    parser.add_argument("--inducing", type=int, default=256)
    parser.add_argument("--inducing-method", choices=("kmeans", "random"), default="kmeans")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--lengthscale", default="auto", help="Positive scalar or 'auto' for median heuristic.")
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--variance-mode", choices=("fitc", "dtc"), default="fitc")
    parser.add_argument("--shuffle-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ssgp-repo",
        type=Path,
        default=None,
        help="Path to thangbui/streaming_sparse_gp for --backend gpflow.",
    )
    parser.add_argument("--gpflow-maxiter", type=int, default=100)
    parser.add_argument("--gpflow-initial-batch", type=int, default=4096)
    return parser.parse_args()


def resolve_lengthscale(value: str, X_train: np.ndarray, seed: int) -> float:
    if value == "auto":
        return median_lengthscale(X_train, seed=seed)
    lengthscale = float(value)
    if lengthscale <= 0.0:
        raise ValueError("--lengthscale must be positive")
    return lengthscale


def run_numpy_backend(args: argparse.Namespace, split_data, lengthscale: float) -> tuple[dict[str, float], dict[str, float]]:
    started = time.perf_counter()
    Z = choose_inducing_points(
        split_data.X_train,
        n_inducing=args.inducing,
        seed=args.model_seed,
        method=args.inducing_method,
    )
    model = FixedInducingStreamingSparseGP(
        inducing_points=Z,
        lengthscale=lengthscale,
        noise_variance=args.noise_std**2,
        kernel_variance=args.kernel_variance,
        jitter=args.jitter,
        variance_mode=args.variance_mode,
    )
    rng = np.random.default_rng(args.seed)
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
    metrics = regression_metrics(split_data.y_test, pred_mean, pred_var, split_data.y_mean, split_data.y_std)
    timing = {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": train_seconds + predict_seconds,
    }
    return metrics, timing


def _gpflow_cov_to_matrix(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance)
    covariance = np.squeeze(covariance)
    if covariance.ndim == 3:
        covariance = covariance[0]
    if covariance.ndim == 1:
        covariance = np.diag(covariance)
    return np.asarray(covariance, dtype=np.float64)


def run_gpflow_backend(args: argparse.Namespace, split_data, lengthscale: float) -> tuple[dict[str, float], dict[str, float]]:
    if args.ssgp_repo is not None:
        code_dir = Path(args.ssgp_repo) / "code"
        if not (code_dir / "osgpr.py").exists():
            raise FileNotFoundError(f"Could not find osgpr.py under {code_dir}")
        sys.path.insert(0, str(code_dir))

    try:
        tf = importlib.import_module("tensorflow")
        gpflow = importlib.import_module("gpflow")
        osgpr = importlib.import_module("osgpr")
    except ImportError as exc:
        raise ImportError(
            "The official SSGP backend needs TensorFlow, GPflow, and thangbui/streaming_sparse_gp. "
            "Clone https://github.com/thangbui/streaming_sparse_gp and pass --ssgp-repo."
        ) from exc

    try:
        gpflow.config.set_default_float(np.float64)
    except Exception:
        pass

    X_train = split_data.X_train.astype(np.float64)
    y_train = split_data.y_train.astype(np.float64)
    X_test = split_data.X_test.astype(np.float64)

    started = time.perf_counter()
    initial_size = min(args.gpflow_initial_batch, X_train.shape[0])
    X_initial = X_train[:initial_size]
    y_initial = y_train[:initial_size]
    Z_old = choose_inducing_points(X_initial, args.inducing, args.model_seed, args.inducing_method)

    kernel = gpflow.kernels.SquaredExponential(
        variance=args.kernel_variance,
        lengthscales=lengthscale,
    )
    initial_model = gpflow.models.SGPR(data=(X_initial, y_initial), kernel=kernel, inducing_variable=Z_old)
    initial_model.likelihood.variance.assign(args.noise_std**2)
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        initial_model.training_loss,
        initial_model.trainable_variables,
        options={"maxiter": args.gpflow_maxiter},
    )
    likelihood_variance = float(initial_model.likelihood.variance.numpy())
    Z_old = initial_model.inducing_variable.Z.numpy()
    mu_old, Su_old = initial_model.predict_f(Z_old, full_cov=True)
    mu_old = np.asarray(mu_old, dtype=np.float64)
    Su_old = _gpflow_cov_to_matrix(Su_old.numpy())
    Kaa_old = kernel(Z_old).numpy()
    current_model = initial_model

    for start in range(initial_size, X_train.shape[0], args.batch_size):
        X_batch = X_train[start : start + args.batch_size]
        y_batch = y_train[start : start + args.batch_size]
        if X_batch.size == 0:
            continue
        Z_new_source = np.vstack([Z_old, X_batch])
        Z_new = choose_inducing_points(Z_new_source, args.inducing, args.model_seed + start, args.inducing_method)
        current_model = osgpr.OSGPR_VFE(
            data=(X_batch, y_batch),
            kernel=kernel,
            mu_old=mu_old,
            Su_old=Su_old,
            Kaa_old=Kaa_old,
            Z_old=Z_old,
            Z=Z_new,
        )
        current_model.likelihood.variance.assign(likelihood_variance)
        optimizer.minimize(
            current_model.training_loss,
            current_model.trainable_variables,
            options={"maxiter": args.gpflow_maxiter},
        )
        likelihood_variance = float(current_model.likelihood.variance.numpy())
        Z_old = current_model.inducing_variable.Z.numpy()
        mu_old, Su_old = current_model.predict_f(Z_old, full_cov=True)
        mu_old = np.asarray(mu_old, dtype=np.float64)
        Su_old = _gpflow_cov_to_matrix(Su_old.numpy())
        Kaa_old = kernel(Z_old).numpy()

    train_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    mean, var = current_model.predict_y(tf.convert_to_tensor(X_test, dtype=gpflow.default_float()))
    pred_mean = mean.numpy()
    pred_var = var.numpy().reshape(-1, 1)
    predict_seconds = time.perf_counter() - predict_started
    metrics = regression_metrics(split_data.y_test, pred_mean, pred_var, split_data.y_mean, split_data.y_std)
    timing = {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": train_seconds + predict_seconds,
    }
    return metrics, timing


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

    if args.backend == "numpy":
        metrics, timing = run_numpy_backend(args, split_data, lengthscale)
    else:
        metrics, timing = run_gpflow_backend(args, split_data, lengthscale)

    payload = base_result_payload("ssgp", args, split_data)
    payload.update(
        {
            "hyperparameters": {
                "backend": args.backend,
                "inducing": args.inducing,
                "inducing_method": args.inducing_method,
                "lengthscale": lengthscale,
                "noise_variance": args.noise_std**2,
                "kernel_variance": args.kernel_variance,
                "variance_mode": args.variance_mode,
            },
            "timing": timing,
            "metrics": metrics,
        }
    )

    output = args.results_dir / f"ssgp_{args.backend}_{timestamp_token()}.json"
    write_json(output, payload)
    print(f"Wrote {output}")
    print(
        "SSGP metrics: "
        f"rmse_orig={metrics['rmse_orig']:.4f}, "
        f"rmse_std={metrics['rmse_std']:.4f}, "
        f"nll_orig={metrics['nll_orig']:.4f}"
    )


if __name__ == "__main__":
    main()
