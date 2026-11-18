from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_data import DATASETS_DIR, new_output, write_csv, write_json
from common import (
    DELAY_COMPONENT_COLUMNS, TARGET_COLUMN, find_airline_table, infer_airline_feature_columns,
    iter_table_chunks, load_csv_splits, read_table_sample,
)


def clean_chunk(chunk: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    required = [TARGET_COLUMN, *columns]
    frame = chunk.loc[:, required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in DELAY_COMPONENT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].fillna(0.0)
    for column in ("Cancelled", "Diverted"):
        if column in frame:
            frame = frame[frame[column].fillna(0.0) == 0.0]
    return frame.dropna(axis=0, subset=required)


def export_csv(raw_path: Path, directory: Path) -> None:
    if any((directory / name).exists() for name in ("train.csv", "test.csv")):
        raise FileExistsError(f"Saved Airline CSVs already exist: {directory}")
    raw = find_airline_table(raw_path)
    columns = infer_airline_feature_columns(read_table_sample(raw))
    total = 0
    mean = np.zeros(len(columns))
    m2 = np.zeros(len(columns))
    for chunk in iter_table_chunks(raw, [TARGET_COLUMN, *columns], 200000):
        frame = clean_chunk(chunk, columns)
        if frame.empty:
            continue
        values = frame[columns].to_numpy(dtype=float)
        count = len(values)
        delta = values.mean(axis=0) - mean
        m2 += values.var(axis=0) * count + delta ** 2 * total * count / (total + count)
        mean += delta * count / (total + count)
        total += count
    if total < 2:
        raise ValueError("Not enough cleaned Airline rows")
    kept = [name for name, std in zip(columns, np.sqrt(m2 / total)) if std > 1e-12]
    if not kept:
        raise ValueError("No nonconstant Airline features")
    cut = int(0.8 * total)
    directory.mkdir(parents=True, exist_ok=True)
    temp_train = directory / "train.csv.part"
    temp_test = directory / "test.csv.part"
    with temp_train.open("w", encoding="utf-8", newline="") as train_handle, temp_test.open("w", encoding="utf-8", newline="") as test_handle:
        headers = [*kept, "target"]
        pd.DataFrame(columns=headers).to_csv(train_handle, index=False)
        pd.DataFrame(columns=headers).to_csv(test_handle, index=False)
        seen = 0
        for chunk in iter_table_chunks(raw, [TARGET_COLUMN, *columns], 200000):
            frame = clean_chunk(chunk, columns)
            values = frame[[*kept, TARGET_COLUMN]].to_numpy(dtype=np.float32).astype(float)
            frame = pd.DataFrame(values, columns=headers)
            n_train = max(0, min(len(frame), cut - seen))
            frame.iloc[:n_train].to_csv(train_handle, index=False, header=False, float_format="%.17g")
            frame.iloc[n_train:].to_csv(test_handle, index=False, header=False, float_format="%.17g")
            seen += len(frame)
    if seen != total:
        raise RuntimeError("Airline source changed during CSV export")
    temp_train.rename(directory / "train.csv")
    temp_test.rename(directory / "test.csv")
    write_json(directory / "config.json", {
        "source": str(raw), "feature_set": "all_numeric", "filter_completed": True,
        "feature_names": kept, "train_rows": cut, "test_rows": total - cut,
        "split": "input order 80/20", "storage_precision": "float32 values in CSV",
    })
    print(f"Saved Airline CSVs: train={cut}, test={total - cut}", flush=True)


def build_commands(args, output: Path) -> list[list[str]]:
    shared = ["--data-dir", str(args.data_dir.resolve()), "--results-dir", str(output.resolve()),
              "--split", "chronological", "--train-fraction", "0.8",
              "--validation-fraction", "0", "--seed", "0", "--model-seed", "37", "--no-shuffle-train"]
    if args.quick:
        shared += ["--train-limit", "128", "--test-limit", "32"]
    commands = []
    for method in dict.fromkeys(args.methods):
        if method == "vorf":
            tail = ["--J", "100", "--mc-test", "1", "--passes", "1", "--mc-train", "1"]
            script = "run_vorf_gp.py"
        elif method == "issgp":
            tail = ["--features", "100", "--batch-size", "1", "--passes", "1"]
            script = "run_issgp.py"
        else:
            tail = ["--backend", "numpy", "--inducing", "100", "--inducing-method", "kmeans",
                    "--batch-size", "1"]
            script = "run_ssgp.py"
        commands.append([sys.executable, str(SCRIPT_DIR / script), *shared, *tail])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 5: Airline models on saved train/test CSVs.")
    parser.add_argument("--data-dir", type=Path, default=DATASETS_DIR / "airline")
    parser.add_argument("--raw-path", type=Path, default=None, help="Export all-numeric Airline CSV splits first")
    parser.add_argument("--prepare-only", action="store_true", help="Export CSVs without training")
    parser.add_argument("--check-data", action="store_true", help="Check saved CSVs without training")
    parser.add_argument("--methods", nargs="+", choices=("issgp", "ssgp", "vorf"), default=["issgp", "ssgp", "vorf"])
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Small smoke run; not table results")
    args = parser.parse_args()
    if args.prepare_only and args.raw_path is None:
        parser.error("--prepare-only requires --raw-path")
    if args.raw_path is not None:
        export_csv(args.raw_path, args.data_dir)
    if args.prepare_only:
        return
    for name in ("train.csv", "test.csv"):
        if not (args.data_dir / name).is_file():
            parser.error(f"Missing {args.data_dir / name}. Supply saved Airline CSVs or use --raw-path <2008.csv>.")
    if args.check_data:
        data = load_csv_splits(args.data_dir)
        print(f"Airline: train={data.raw_train_size}, test={data.raw_test_size}, features={len(data.feature_names)}")
        return
    output = new_output(SCRIPT_DIR, args.results_dir)
    commands = build_commands(args, output)
    write_json(output / "config.json", {"experiment": 4, "args": vars(args), "commands": commands,
                                        "feature_set": "all_numeric", "evaluation_mc": 1})
    rows = []
    for command in commands:
        print(f"Running {Path(command[1]).name}", flush=True)
        subprocess.run(command, cwd=SCRIPT_DIR.parents[1], check=True)
        rows = []
        for path in sorted(output.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("metrics"):
                rows.append({"method": payload["method"], "quick": args.quick,
                             "rmse_std": payload["metrics"]["rmse_std"],
                             "nll_orig": payload["metrics"]["nll_orig"], "file": path.name})
        write_csv(output / "summary.csv", rows)
        write_csv(output / "table5.csv", [{"method": r["method"], "rmse_std": f"{r['rmse_std']:.4f}",
                                          "mnll": f"{r['nll_orig']:.4f}"} for r in rows])
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
