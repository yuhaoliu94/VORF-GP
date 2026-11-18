from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import time
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_NPZ_PATH,
    DEFAULT_RESULTS_DIR,
    add_src_to_path,
    base_result_payload,
    make_splits,
    regression_metrics,
    timestamp_token,
    write_json,
)


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
add_src_to_path()

from orf_gp.config import GPConfig  # noqa: E402
from orf_gp.models import OnlineGP  # noqa: E402
from orf_gp.utils import set_global_seeds  # noqa: E402


class SilentMonitor:
    def check(self, model: OnlineGP, losses: tuple[float, ...]) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project VORF/ORF-GP model on prepared Airline data.")
    parser.add_argument("--data-npz", "--data-dir", dest="data_npz", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=37)
    parser.add_argument("--J", type=int, default=512)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--mc-train", type=int, default=1)
    parser.add_argument("--mc-test", type=int, default=100)
    parser.add_argument("--mc-chunk-size", type=int, default=5)
    parser.add_argument("--shuffle-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rff-model", choices=("Orthogonal", "RBF", "Maxwell"), default="Orthogonal")
    parser.add_argument("--sigma-l", type=float, default=1.0)
    parser.add_argument("--gamma-obs", type=float, default=1.0)
    parser.add_argument("--beta-kl-scale", type=float, default=1e-4)
    parser.add_argument("--beta-ortho-scale", type=float, default=1e-3)
    parser.add_argument("--lr-mu", type=float, default=0.005)
    parser.add_argument("--lr-b", type=float, default=0.0001)
    parser.add_argument("--lr-l", type=float, default=0.03)
    parser.add_argument("--lr-sigma", type=float, default=0.03)
    parser.add_argument("--init-gamma", type=float, default=None)
    parser.add_argument("--init-log-l", type=float, default=None)
    parser.add_argument("--init-sigma", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--verbose-train", action="store_true")
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Save the trained VORF-GP model checkpoint after training.",
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        default=None,
        help="Load a trained VORF-GP checkpoint and evaluate it without retraining.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train/save only; skip final test evaluation.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace, dx: int, dy: int) -> GPConfig:
    return GPConfig(
        rff_model=args.rff_model,
        model_seed=args.model_seed,
        J=args.J,
        Dx=dx,
        Dy=dy,
        sigma_l=args.sigma_l,
        gamma_obs=args.gamma_obs,
        MC=args.mc_train,
        beta_kl=args.beta_kl_scale / dx,
        beta_ortho=args.beta_ortho_scale / args.J,
        meta_lr={
            "mu": args.lr_mu,
            "b": args.lr_b,
            "l": args.lr_l,
            "sigma": args.lr_sigma,
        },
        init_gamma=args.init_gamma,
        init_log_l=args.init_log_l,
        init_sigma=args.init_sigma,
        alpha_meta=0.01,
        fixed_step=-1,
        log_interval=args.log_interval,
        verbal=args.verbose_train,
    )


def save_checkpoint(
    path: Path,
    model: OnlineGP,
    config: GPConfig,
    args: argparse.Namespace,
    split_data,
    train_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "vorf_gp_experiment4_checkpoint_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "config": config,
        "train_seconds": train_seconds,
        "args": vars(args),
        "data": {
            "source": split_data.source,
            "feature_names": split_data.feature_names,
            "n_features": len(split_data.feature_names),
            "train_size": split_data.raw_train_size,
            "validation_size": split_data.raw_val_size,
            "test_size": split_data.raw_test_size,
            "x_mean": split_data.x_mean,
            "x_std": split_data.x_std,
            "y_mean": split_data.y_mean,
            "y_std": split_data.y_std,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "vorf_gp_experiment4_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return payload


def evaluate_vorf_model(model: OnlineGP, split_data, args: argparse.Namespace) -> tuple[dict[str, float], float]:
    predict_started = time.perf_counter()
    pred_mean, pred_var, student_t_nll, model_mse, model_rmse = model.observation_layer.predict_batch_mc(
        split_data.X_test,
        split_data.y_test,
        MC=args.mc_test,
        chunk_size=args.mc_chunk_size,
    )
    predict_seconds = time.perf_counter() - predict_started
    metrics = regression_metrics(
        split_data.y_test,
        pred_mean,
        pred_var,
        y_mean=split_data.y_mean,
        y_std=split_data.y_std,
    )
    metrics.update(
        {
            "student_t_nll_std": float(student_t_nll),
            "model_mse_std": float(model_mse),
            "model_rmse_std": float(model_rmse),
        }
    )
    return metrics, predict_seconds


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

    if args.checkpoint_in is not None:
        checkpoint = load_checkpoint(args.checkpoint_in)
        model = checkpoint["model"]
        config = checkpoint["config"]
        train_seconds = float(checkpoint.get("train_seconds", 0.0))
        print(f"Loaded VORF-GP checkpoint from {args.checkpoint_in}")
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            set_global_seeds(args.model_seed)

        config = make_config(args, dx=split_data.X_train.shape[1], dy=split_data.y_train.shape[1])
        model = OnlineGP(config=config, monitor=SilentMonitor())
        model.initialize_structure()

        rng = np.random.default_rng(args.seed)
        started = time.perf_counter()
        for pass_index in range(args.passes):
            order = np.arange(split_data.X_train.shape[0])
            if args.shuffle_train:
                rng.shuffle(order)
            print(f"VORF-GP pass {pass_index + 1}/{args.passes}, steps={order.size}")
            for local_step, index in enumerate(order, start=1):
                model.step(split_data.y_train[index], split_data.X_train[index], MC=args.mc_train)
                if args.verbose_train and local_step % args.log_interval == 0:
                    print(f"  step {local_step}/{order.size}")
        train_seconds = time.perf_counter() - started

        if args.checkpoint_out is not None:
            save_checkpoint(args.checkpoint_out, model, config, args, split_data, train_seconds)
            print(f"Saved VORF-GP checkpoint to {args.checkpoint_out}")

    metrics = None
    predict_seconds = 0.0
    if not args.skip_eval:
        metrics, predict_seconds = evaluate_vorf_model(model, split_data, args)

    payload = base_result_payload("vorf_gp", args, split_data)
    payload.update(
        {
            "hyperparameters": {
                "J": args.J,
                "rff_model": args.rff_model,
                "meta_lr": config.meta_lr,
                "beta_kl": config.beta_kl,
                "beta_ortho": config.beta_ortho,
                "passes": args.passes,
                "mc_train": args.mc_train,
                "mc_test": args.mc_test,
            },
            "checkpoint": {
                "checkpoint_in": args.checkpoint_in,
                "checkpoint_out": args.checkpoint_out,
                "skip_eval": args.skip_eval,
            },
            "timing": {
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": train_seconds + predict_seconds,
            },
            "metrics": metrics or {},
        }
    )

    output = args.results_dir / f"vorf_gp_{timestamp_token()}.json"
    write_json(output, payload)
    print(f"Wrote {output}")
    if metrics is not None:
        print(
            "VORF-GP metrics: "
            f"rmse_orig={metrics['rmse_orig']:.4f}, "
            f"rmse_std={metrics['rmse_std']:.4f}, "
            f"nll_orig={metrics['nll_orig']:.4f}"
        )
    else:
        print("VORF-GP evaluation skipped.")


if __name__ == "__main__":
    main()
