from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS = ("concrete", "kin8nm", "powerplant", "protein")

#RNG seeds (randomly generated) for random features and training order;
SPLIT_STATES = {
    "concrete": (12, 14, 5, 16, 1),
    "kin8nm": (5, 22, 16, 29, 18),
    "powerplant": (14, 28, 13, 16, 4),
    "protein": (0, 11, 12, 7, 14),
}


@dataclass
class SavedSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    directory: Path


def read_split(directory: Path) -> SavedSplit:
    frames = []
    for name in ("train.csv", "test.csv"):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved data: {path}")
        frame = pd.read_csv(path, float_precision="round_trip")
        if frame.empty or len(frame.columns) < 2 or frame.columns[-1] != "target":
            raise ValueError(f"Expected feature columns followed by target: {path}")
        if frame.columns.duplicated().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid numeric data: {path}")
        frames.append(frame)
    train, test = frames
    if list(train.columns) != list(test.columns):
        raise ValueError(f"Train/test columns differ: {directory}")
    return SavedSplit(
        np.ascontiguousarray(train.iloc[:, :-1].to_numpy(dtype=float)),
        np.ascontiguousarray(train.iloc[:, -1:].to_numpy(dtype=float)),
        np.ascontiguousarray(test.iloc[:, :-1].to_numpy(dtype=float)),
        np.ascontiguousarray(test.iloc[:, -1:].to_numpy(dtype=float)),
        list(train.columns[:-1]),
        directory.resolve(),
    )


def split_record(data: SavedSplit) -> dict:
    return {
        "directory": str(data.directory),
        "train_rows": len(data.x_train),
        "test_rows": len(data.x_test),
        "features": data.feature_names,
        "sha256": {
            name: hashlib.sha256((data.directory / name).read_bytes()).hexdigest()
            for name in ("train.csv", "test.csv")
        },
    }


def normalize(data: SavedSplit, epsilon: float = 0.0) -> dict:
    x_std = data.x_train.std(axis=0)
    y_std = data.y_train.std(axis=0)
    x_std = x_std + epsilon if epsilon else np.maximum(x_std, 1e-12)
    y_std = y_std + epsilon if epsilon else np.maximum(y_std, 1e-12)
    x_mean = data.x_train.mean(axis=0)
    y_mean = data.y_train.mean(axis=0)
    return {
        "x_train": (data.x_train - x_mean) / x_std,
        "y_train": (data.y_train - y_mean) / y_std,
        "x_test": (data.x_test - x_mean) / x_std,
        "y_test": (data.y_test - y_mean) / y_std,
        "y_std": y_std,
        "y_test_std": np.maximum(data.y_test.std(axis=0), 1e-12),
    }


def training_stream(data: SavedSplit, seed: int, repeats: int, validation_fraction: float, epsilon: float = 0.0) -> dict:
    result = normalize(data, epsilon)
    rng = np.random.RandomState(seed)
    outer_order = np.arange(len(data.x_train) + len(data.x_test))
    rng.shuffle(outer_order)
    x_train, y_train = result["x_train"], result["y_train"]
    x_val = y_val = None
    if validation_fraction:
        order = np.arange(len(x_train))
        rng.shuffle(order)
        cut = int(len(order) * (1.0 - validation_fraction))
        if not 0 < cut < len(order):
            raise ValueError("Not enough training rows for validation")
        x_val, y_val = x_train[order[cut:]], y_train[order[cut:]]
        x_train, y_train = x_train[order[:cut]], y_train[order[:cut]]
    result["train_base_size"] = len(x_train)
    result["x_val"], result["y_val"] = x_val, y_val
    x_train = np.repeat(x_train, repeats, axis=0)
    y_train = np.repeat(y_train, repeats, axis=0)
    order = np.arange(len(x_train))
    rng.shuffle(order)
    result["x_train"], result["y_train"] = x_train[order], y_train[order]
    return result


def add_tabular_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DATASETS_DIR, help="Saved CSV root")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--copies", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--results-dir", type=Path, default=None, help="New output directory")
    parser.add_argument("--check-data", action="store_true", help="Check CSVs without training")
    parser.add_argument("--quick", action="store_true", help="Small smoke run; not table results")


def load_tabular(args: argparse.Namespace) -> dict[tuple[str, int], SavedSplit]:
    loaded = {}
    for dataset in dict.fromkeys(args.datasets):
        for copy in dict.fromkeys(args.copies):
            data = read_split(args.data_dir / f"{dataset}_{copy}")
            loaded[dataset, copy] = data
            print(f"{dataset}_{copy}: train={len(data.x_train)}, test={len(data.x_test)}, features={len(data.feature_names)}", flush=True)
    return loaded


def new_output(script_dir: Path, requested: Path | None) -> Path:
    output = requested or script_dir / "results_saved_data" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_tables(output: Path, rows: list[dict], table_numbers: tuple[int, ...]) -> None:
    write_csv(output / "raw_results.csv", rows)
    summary = []
    for method, dataset in dict.fromkeys((r["method"], r["dataset"]) for r in rows):
        group = [r for r in rows if r["method"] == method and r["dataset"] == dataset]
        row = {"method": method, "dataset": dataset, "n": len(group)}
        for metric in ("rmse_std", "nll_orig"):
            values = [r[metric] for r in group if metric in r]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
        summary.append(row)
    write_csv(output / "summary.csv", summary)
    for number in table_numbers:
        metric = "nll_orig" if number == 3 else "rmse_std"
        matrix = []
        for method in dict.fromkeys(r["method"] for r in summary):
            row = {"method": method}
            for item in summary:
                if item["method"] == method and f"{metric}_mean" in item:
                    row[item["dataset"]] = f"{item[f'{metric}_mean']:.2f} ({item[f'{metric}_std']:.2f})"
            matrix.append(row)
        write_csv(output / f"table{number}.csv", matrix)
