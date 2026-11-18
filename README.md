# VORF-GP

Code for the VORF-GP experiments. The model is in `src/orf_gp`, the ORF/SORF baselines are in `benchmark/`, and the fixed train/test splits are in `datasets/`.

## Setup

Tested with Python 3.11, numpy 2.4, scipy 1.17, pandas 3.0, scikit-learn 1.8, matplotlib 3.10, and xlrd 2.0.

```bash
pip install numpy scipy pandas scikit-learn matplotlib xlrd jupyter
```

## Experiment 1: Concrete, varying J (Table 1)

Open `experiments/experiment 1/` and run the notebooks `01_` to `05_` in order. `04_Compare_J_seed.ipynb` produces `MNLL.png` and `RMSE.png`, and `05_feature_concrete.ipynb` prints Table 1.

## Experiment 2: VORF vs. ORF vs. SORF (Table 2)

Run from the repository root:

```bash
python "experiments/experiment 2/run_saved_data.py"
```

Results are written to `experiments/experiment 2/results_saved_data/<timestamp>/table2.csv`.

## Experiment 3: VORF vs. i.i.d. RBF vs. i.i.d. Maxwell (Tables 3 and 4)

Run from the repository root:

```bash
python "experiments/experiment 3/run_saved_data.py"
```

Results are written to `experiments/experiment 3/results_saved_data/<timestamp>/table3.csv` (MNLL) and `table4.csv` (RMSE).

## Experiment 4: Airline (Table 5)

The Airline data is not in the repository. Download the 2008 Airline on-time data (`2008.csv`), then run from the repository root:

```bash
# 1. Export train/test CSVs to datasets/airline/ (only once)
python "experiments/experiment 4/run_saved_data.py" --raw-path path/to/2008.csv --prepare-only

# 2. Train and evaluate ISSGP, SSGP and VORF
python "experiments/experiment 4/run_saved_data.py"
```

Results are written to `experiments/experiment 4/results_saved_data/<timestamp>/table5.csv`.
