"""Seed and paths shared by every notebook/script. Works locally and on Kaggle."""
from pathlib import Path

SEED = 42
N_FOLDS = 5

TARGETS = {"regression": "price", "classification": "vehicle_class"}
ID_COLS = {"regression": "id", "classification": "vehicle_id"}

KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.is_dir()

# Local: repo root (parent of src/). Kaggle: outputs must go to /kaggle/working.
ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else ROOT


def _find_kaggle_data_dir(task: str) -> Path:
    # Kaggle mounts competition data at /kaggle/input/<slug>/ and the slug isn't known
    # in advance, so pick the folder whose train.csv contains this task's target column.
    target = TARGETS[task]
    for train in KAGGLE_INPUT.rglob("*train*.csv"):
        with open(train) as f:
            if target in f.readline().strip().split(","):
                return train.parent
    raise FileNotFoundError(f"No train csv with column '{target}' under {KAGGLE_INPUT}")


def data_dir(task: str) -> Path:
    if task not in TARGETS:
        raise ValueError(f"task must be one of {list(TARGETS)}")
    return _find_kaggle_data_dir(task) if ON_KAGGLE else ROOT / "dataset" / task


def results_dir(task: str) -> Path:
    d = OUT_ROOT / "results" / task
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def submissions_dir(task: str) -> Path:
    d = OUT_ROOT / "submissions" / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(task: str):
    """Return (train, test, sample_submission) DataFrames for 'regression' or 'classification'."""
    import pandas as pd

    d = data_dir(task)
    # Kaggle may keep the original download names (e.g. vehicle_train_10000.csv); match by prefix.
    def pick(kind):
        exact = d / f"{kind}.csv"
        return exact if exact.exists() else next(p for p in d.glob("*.csv") if kind in p.stem)

    return (pd.read_csv(pick("train")), pd.read_csv(pick("test")),
            pd.read_csv(pick("sample_submission")))
