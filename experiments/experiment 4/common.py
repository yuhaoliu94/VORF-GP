from __future__ import annotations

import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
AIRLINE_DATA_DIR = PROJECT_ROOT / "data" / "airline"
DEFAULT_NPZ_PATH = AIRLINE_DATA_DIR / "airline_numeric.npz"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

TARGET_COLUMN = "ArrDelay"

AIRLINE_NUMERIC_COLUMNS = (
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "DepTime",
    "CRSDepTime",
    "ArrTime",
    "CRSArrTime",
    "FlightNum",
    "ActualElapsedTime",
    "CRSElapsedTime",
    "AirTime",
    "DepDelay",
    "Distance",
    "TaxiIn",
    "TaxiOut",
    "Cancelled",
    "Diverted",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
)

DELAY_COMPONENT_COLUMNS = {
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
}


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None
    y_val: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    feature_names: list[str]
    source: str
    raw_train_size: int
    raw_val_size: int
    raw_test_size: int


def add_src_to_path() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def timestamp_token() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def first_csv_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            name
            for name in zf.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in {".csv", ".txt"}
        ]
    if not names:
        raise FileNotFoundError(f"No CSV/TXT member found inside {zip_path}")
    names.sort(key=lambda name: ("2008" not in Path(name).name.lower(), len(name)))
    return names[0]


def find_airline_table(raw_path: Path | None) -> Path:
    if raw_path is None:
        raw_path = AIRLINE_DATA_DIR
    raw_path = Path(raw_path)
    if raw_path.is_file():
        return raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Airline raw path does not exist: {raw_path}. "
            "Download the Kaggle dataset into data/airline first, or pass --raw-path."
        )

    candidates: list[Path] = []
    for pattern in ("*.csv", "*.csv.gz", "*.zip", "*.txt"):
        candidates.extend(raw_path.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No CSV, CSV.GZ, TXT, or ZIP files found under {raw_path}")

    def sort_key(path: Path) -> tuple[int, int, int]:
        name = path.name.lower()
        likely = int("2008" not in name and "flight" not in name and "airline" not in name)
        zipped = int(path.suffix.lower() != ".zip")
        return likely, zipped, -path.stat().st_size

    return sorted(candidates, key=sort_key)[0]


def read_table_sample(path: Path, nrows: int = 5000) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        member = first_csv_member(path)
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as handle:
                return pd.read_csv(handle, nrows=nrows, low_memory=False)
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def iter_table_chunks(path: Path, usecols: list[str], chunksize: int) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        member = first_csv_member(path)
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as handle:
                yield from pd.read_csv(handle, usecols=usecols, chunksize=chunksize, low_memory=False)
        return
    yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)


def infer_airline_feature_columns(sample: pd.DataFrame) -> list[str]:
    columns_by_lower = {column.lower(): column for column in sample.columns}

    if TARGET_COLUMN.lower() not in columns_by_lower:
        raise ValueError(f"Target column {TARGET_COLUMN!r} was not found. Columns: {list(sample.columns)}")

    feature_names = [columns_by_lower[name.lower()] for name in AIRLINE_NUMERIC_COLUMNS
                     if name.lower() in columns_by_lower]
    if not feature_names:
        raise ValueError("No numeric Airline feature columns found")
    return feature_names


def load_npz_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    with np.load(path, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1, 1)
        feature_names = [str(name) for name in data["feature_names"].tolist()]
        source = str(data["source"].item()) if "source" in data.files else str(path)
    return X, y, feature_names, source


def load_csv_splits(path: Path, train_limit: int | None = None, test_limit: int | None = None) -> SplitData:
    frames = []
    for name, limit in (("train.csv", train_limit), ("test.csv", test_limit)):
        if limit is not None and limit < 1:
            raise ValueError("Row limits must be positive")
        frame = pd.read_csv(path / name, nrows=limit, float_precision="round_trip")
        if frame.empty or len(frame.columns) < 2 or frame.columns[-1] not in {"target", TARGET_COLUMN}:
            raise ValueError(f"Expected feature columns followed by target: {path / name}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite data: {path / name}")
        frames.append(frame)
    train, test = frames
    if list(train.columns) != list(test.columns):
        raise ValueError("Airline train/test columns differ")
    x_train = np.ascontiguousarray(train.iloc[:, :-1].to_numpy(dtype=float))
    y_train = np.ascontiguousarray(train.iloc[:, -1:].to_numpy(dtype=float))
    x_test = np.ascontiguousarray(test.iloc[:, :-1].to_numpy(dtype=float))
    y_test = np.ascontiguousarray(test.iloc[:, -1:].to_numpy(dtype=float))
    x_mean = x_train.mean(axis=0)
    x_std = np.maximum(x_train.std(axis=0), 1e-12)
    y_mean = y_train.mean(axis=0)
    y_std = np.maximum(y_train.std(axis=0), 1e-12)
    return SplitData(
        X_train=(x_train - x_mean) / x_std, y_train=(y_train - y_mean) / y_std,
        X_val=None, y_val=None,
        X_test=(x_test - x_mean) / x_std, y_test=(y_test - y_mean) / y_std,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        feature_names=list(train.columns[:-1]), source=str(path.resolve()),
        raw_train_size=len(train), raw_val_size=0, raw_test_size=len(test),
    )


def make_splits(
    data_npz: Path,
    train_fraction: float,
    validation_fraction: float,
    split: str,
    seed: int,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> SplitData:
    if Path(data_npz).is_dir():
        if validation_fraction != 0.0 or split != "chronological" or train_fraction != 0.8:
            raise ValueError("Saved CSV splits are fixed; use chronological, train_fraction=0.8, validation_fraction=0")
        return load_csv_splits(Path(data_npz), train_limit, test_limit)
    X, y, feature_names, source = load_npz_dataset(data_npz)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if split not in {"chronological", "random"}:
        raise ValueError("--split must be chronological or random")

    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    if split == "random":
        rng.shuffle(indices)

    n_train_total = max(1, int(X.shape[0] * train_fraction))
    train_indices = indices[:n_train_total]
    test_indices = indices[n_train_total:]
    if test_indices.size == 0:
        raise ValueError("The split produced an empty test set; use more rows or a smaller train fraction.")

    if validation_fraction > 0.0:
        n_val = max(1, int(train_indices.size * validation_fraction))
        n_train = max(1, train_indices.size - n_val)
        if split == "random":
            train_pool = train_indices
            rng.shuffle(train_pool)
            val_indices = train_pool[n_train:]
            train_indices = train_pool[:n_train]
        else:
            val_indices = train_indices[n_train:]
            train_indices = train_indices[:n_train]
    else:
        val_indices = np.array([], dtype=int)

    if train_limit is not None:
        train_indices = train_indices[:train_limit]
    if test_limit is not None:
        test_indices = test_indices[:test_limit]

    X_train_raw = X[train_indices]
    y_train_raw = y[train_indices]
    X_val_raw = X[val_indices] if val_indices.size else None
    y_val_raw = y[val_indices] if val_indices.size else None
    X_test_raw = X[test_indices]
    y_test_raw = y[test_indices]

    x_mean = X_train_raw.mean(axis=0)
    x_std = np.maximum(X_train_raw.std(axis=0), 1e-12)
    y_mean = y_train_raw.mean(axis=0)
    y_std = np.maximum(y_train_raw.std(axis=0), 1e-12)

    X_train = (X_train_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std
    X_test = (X_test_raw - x_mean) / x_std
    y_test = (y_test_raw - y_mean) / y_std
    X_val = None if X_val_raw is None else (X_val_raw - x_mean) / x_std
    y_val = None if y_val_raw is None else (y_val_raw - y_mean) / y_std

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=feature_names,
        source=source,
        raw_train_size=int(train_indices.size),
        raw_val_size=int(val_indices.size),
        raw_test_size=int(test_indices.size),
    )


def iter_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    order = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(order)
    for start in range(0, order.size, batch_size):
        batch_indices = order[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]


def median_lengthscale(X: np.ndarray, seed: int, max_points: int = 2000) -> float:
    if X.shape[0] < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    sample_size = min(max_points, X.shape[0])
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    sample = X[sample_indices]
    distances = pdist(sample, metric="euclidean")
    distances = distances[np.isfinite(distances) & (distances > 1e-12)]
    if distances.size == 0:
        return 1.0
    return float(np.median(distances))


def rbf_kernel(X: np.ndarray, Z: np.ndarray, lengthscale: float, variance: float = 1.0) -> np.ndarray:
    X_scaled = X / lengthscale
    Z_scaled = Z / lengthscale
    x2 = np.sum(X_scaled * X_scaled, axis=1, keepdims=True)
    z2 = np.sum(Z_scaled * Z_scaled, axis=1, keepdims=True).T
    sqdist = np.maximum(x2 + z2 - 2.0 * X_scaled @ Z_scaled.T, 0.0)
    return variance * np.exp(-0.5 * sqdist)


def choose_inducing_points(
    X: np.ndarray,
    n_inducing: int,
    seed: int,
    method: str = "kmeans",
) -> np.ndarray:
    if n_inducing >= X.shape[0]:
        return np.array(X, copy=True)
    rng = np.random.default_rng(seed)
    if method == "random":
        indices = rng.choice(X.shape[0], size=n_inducing, replace=False)
        return np.array(X[indices], copy=True)
    if method != "kmeans":
        raise ValueError("--inducing-method must be kmeans or random")
    from sklearn.cluster import MiniBatchKMeans

    kwargs = {
        "n_clusters": n_inducing,
        "random_state": seed,
        "batch_size": max(1024, 4 * n_inducing),
        "max_iter": 100,
    }
    try:
        model = MiniBatchKMeans(n_init="auto", **kwargs)
    except TypeError:
        model = MiniBatchKMeans(n_init=3, **kwargs)
    model.fit(X)
    return np.asarray(model.cluster_centers_, dtype=np.float64)


def regression_metrics(
    y_true_std: np.ndarray,
    y_mean_pred_std: np.ndarray,
    y_var_pred_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, float]:
    y_true_std = y_true_std.reshape(-1, 1)
    y_mean_pred_std = y_mean_pred_std.reshape(-1, 1)
    y_var_pred_std = np.maximum(y_var_pred_std.reshape(-1, 1), 1e-12)

    residual_std = y_true_std - y_mean_pred_std
    mse_std = float(np.mean(residual_std**2))
    rmse_std = float(np.sqrt(mse_std))
    mae_std = float(np.mean(np.abs(residual_std)))
    nll_std = float(
        np.mean(0.5 * (np.log(2.0 * np.pi * y_var_pred_std) + residual_std**2 / y_var_pred_std))
    )

    y_true_orig = y_true_std * y_std + y_mean
    y_pred_orig = y_mean_pred_std * y_std + y_mean
    y_var_orig = y_var_pred_std * (y_std**2)
    residual_orig = y_true_orig - y_pred_orig
    mse_orig = float(np.mean(residual_orig**2))
    rmse_orig = float(np.sqrt(mse_orig))
    mae_orig = float(np.mean(np.abs(residual_orig)))
    nll_orig = float(
        np.mean(0.5 * (np.log(2.0 * np.pi * y_var_orig) + residual_orig**2 / y_var_orig))
    )

    return {
        "mse_std": mse_std,
        "rmse_std": rmse_std,
        "mae_std": mae_std,
        "nll_std": nll_std,
        "mse_orig": mse_orig,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "nll_orig": nll_orig,
    }


def base_result_payload(method: str, args: Any, split_data: SplitData) -> dict[str, Any]:
    return {
        "method": method,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "data": {
            "source": split_data.source,
            "feature_names": split_data.feature_names,
            "n_features": len(split_data.feature_names),
            "train_size": split_data.raw_train_size,
            "validation_size": split_data.raw_val_size,
            "test_size": split_data.raw_test_size,
            "y_mean": split_data.y_mean,
            "y_std": split_data.y_std,
        },
    }
