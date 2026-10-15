"""One CV routine and one submission writer, shared by every model, so all scores are comparable.

cross_validate() scores a model with the same folds every time and appends a row to
results/<task>/experiments.csv. That file becomes the report's model-comparison table.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from src import config


def _regression_metrics(y_true, y_pred):
    # RMSLE is undefined for negative predictions (a raw-price linear model can produce them),
    # so predictions are clipped at 1 rupee for that metric only.
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmsle": root_mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 1, None))),
    }


def _classification_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


TASK_SETUP = {
    "regression": (KFold, _regression_metrics),
    "classification": (StratifiedKFold, _classification_metrics),
}
# The competition metric per task; also reported on the training folds to show over/underfitting.
PRIMARY = {"regression": "r2", "classification": "accuracy"}


def _splitter(task):
    return TASK_SETUP[task][0](n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)


def log_target(model):
    """Train on log(price), predict back in rupees. Wraps any regressor."""
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)


def cross_validate(model, X, y, task, name, params=None, notes="", log=True, extra=None):
    """K-fold CV (stratified for classification) with config.SEED. Returns per-fold scores as a DataFrame.

    The model is cloned and re-fitted on each fold's training part, so anything it learns from data
    (imputation medians, scaling, encoders) never sees that fold's validation rows.
    `extra`: additional columns for the log row (e.g. noisy-CV accuracies).
    """
    metric_fn = TASK_SETUP[task][1]
    primary = PRIMARY[task]
    X, y = X.reset_index(drop=True), pd.Series(np.asarray(y))

    rows = []
    for fold, (tr, va) in enumerate(_splitter(task).split(X, y)):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        scores = metric_fn(y.iloc[va], fitted.predict(X.iloc[va]))
        # Same metric on the rows the model trained on: a big train-vs-CV gap means overfitting.
        scores[f"train_{primary}"] = metric_fn(y.iloc[tr], fitted.predict(X.iloc[tr]))[primary]
        rows.append({"fold": fold, **scores})
    folds = pd.DataFrame(rows).set_index("fold")

    if log:
        summary = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": name,
                   "params": json.dumps(params or {}), "n_rows": len(X)}
        for m in folds.columns:
            summary[f"{m}_mean"] = folds[m].mean()
            summary[f"{m}_std"] = folds[m].std()
        summary.update(extra or {})
        summary.update({"notes": notes, "public_lb": ""})
        path = config.results_dir(task) / "experiments.csv"
        log_df = pd.DataFrame([summary])
        if path.exists():
            # Re-running a notebook replaces that experiment's row instead of duplicating it
            # (a public_lb value typed in by hand is carried over).
            old = pd.read_csv(path, dtype={"public_lb": str}, keep_default_na=False)
            same = (old["model"] == name) & (old["params"] == summary["params"])
            if same.any():
                log_df["public_lb"] = old.loc[same, "public_lb"].iloc[-1]
            log_df = pd.concat([old[~same], log_df], ignore_index=True)
        log_df.to_csv(path, index=False)
    return folds


def grid_search(make_model, grid, X, y, task, name, notes=""):
    """Try every parameter combination in `grid` (list of dicts) on the same CV folds, log only the best
    one (by the task's primary metric) to experiments.csv, and return all results as a table.

    Note: picking the best of several settings on the same folds makes that best score slightly
    optimistic; with a handful of settings and 15k rows the effect is small."""
    primary = PRIMARY[task]
    results = []
    for params in grid:
        folds = cross_validate(make_model(**params), X, y, task, name, log=False)
        results.append({**params, f"{primary}_mean": folds[primary].mean(), f"{primary}_std": folds[primary].std(),
                        f"train_{primary}": folds[f"train_{primary}"].mean()})
    table = pd.DataFrame(results)
    # Take the best params from the original dicts: reading them back from the table would turn
    # e.g. max_depth 12 into 12.0 (pandas makes a column float when it also holds None).
    best = grid[int(table[f"{primary}_mean"].idxmax())]
    table = table.sort_values(f"{primary}_mean", ascending=False).reset_index(drop=True)
    cross_validate(make_model(**best), X, y, task, name, params=best,
                   notes=(notes + f" | best of {len(grid)} settings").strip(" |"))
    return table, best


def oof_predict(model, X, y, task):
    """Out-of-fold predictions: each row is predicted by the model trained on the other folds
    (same folds as cross_validate). Useful for looking at *where* a model makes its errors."""
    X, y = X.reset_index(drop=True), pd.Series(np.asarray(y))
    pred = np.empty(len(X), dtype=float if task == "regression" else object)
    for tr, va in _splitter(task).split(X, y):
        pred[va] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return pred


def noise_curve(model, X, y, task, ks=(0, 0.05, 0.1, 0.2, 0.3), seed=config.SEED):
    """Robustness check for the classification task, whose hidden test set has extra measurement noise.

    Same folds as cross_validate(). The model is trained on the clean training part; then the validation
    features get Gaussian noise with std = k x (that feature's std in the training part), for each k.
    Returns one row per k with the mean and std of the primary metric over folds.
    k = 0 is ordinary CV."""
    metric_fn, primary = TASK_SETUP[task][1], PRIMARY[task]
    X, y = X.reset_index(drop=True), pd.Series(np.asarray(y))
    rows = []
    for fold, (tr, va) in enumerate(_splitter(task).split(X, y)):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        sd = X.iloc[tr].std().to_numpy()
        rng = np.random.default_rng(seed + fold)  # same noise draw for every model -> fair comparison
        base = rng.standard_normal((len(va), X.shape[1]))
        for k in ks:
            Xv = X.iloc[va] + base * sd * k
            rows.append({"k": k, "fold": fold, primary: metric_fn(y.iloc[va], fitted.predict(Xv))[primary]})
    return pd.DataFrame(rows).groupby("k")[primary].agg(["mean", "std"])


def summarize(folds: pd.DataFrame) -> pd.Series:
    """'mean ± std' per metric, for printing."""
    return folds.agg(["mean", "std"]).T.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)


def make_submission(model, X_train, y_train, X_test, test_ids, sample, task, name, date=None):
    """Fit on all training rows, predict test, check the format against sample_submission, save the csv.
    `date` (YYYY-MM-DD) fixes the file-name prefix so a re-run recreates the file that was uploaded."""
    fitted = clone(model).fit(X_train, y_train)
    id_col, target_col = sample.columns
    sub = pd.DataFrame({id_col: np.asarray(test_ids), target_col: fitted.predict(X_test)})

    assert sub.shape == sample.shape, f"shape {sub.shape} != sample {sample.shape}"
    assert (sub[id_col].values == sample[id_col].values).all(), "ids differ from sample_submission order"
    assert sub[target_col].notna().all(), "missing predictions"

    path = config.submissions_dir(task) / f"{date or f'{datetime.now():%Y-%m-%d}'}_{name}.csv"
    sub.to_csv(path, index=False)
    return path, sub
