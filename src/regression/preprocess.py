"""Preprocessing for the flight-price regression task.

Two kinds of steps:
- Fixed rules that learn nothing from the data's prices, so they can run once before CV:
  airline recovery from the flight-code prefix, dropping exact duplicate rows.
- Steps that learn values from data (duration medians, scaling, encoders). These live inside
  an sklearn Pipeline so each CV fold re-fits them on its own training part only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config

TASK = "regression"
TARGET = config.TARGETS[TASK]
ID = config.ID_COLS[TASK]

STOPS = ["zero", "one", "two_or_more"]
NUMERIC = ["duration", "days_left"]
CATEGORICAL = ["source_city", "destination_city", "departure_time", "arrival_time", "flight_class"]
DURATION_KEYS = ["flight", "source_city", "destination_city", "stops"]
DURATION_FALLBACK_KEYS = ["source_city", "destination_city", "stops"]


# ---------- fixed rules (run once, before CV) ----------

def flight_prefix(flight: pd.Series) -> pd.Series:
    """'6E 285' -> '6E'."""
    return flight.str.split().str[0]


def build_prefix_map(*frames: pd.DataFrame) -> pd.Series:
    """prefix -> airline, from rows where airline is known. Uses flight codes only, never prices."""
    known = pd.concat([f[["flight", "airline"]] for f in frames]).dropna()
    pairs = known.assign(prefix=flight_prefix(known["flight"])).drop_duplicates(["prefix", "airline"])
    conflicts = pairs["prefix"].duplicated(keep=False)
    if conflicts.any():
        raise ValueError(f"prefixes mapping to >1 airline: {sorted(pairs.loc[conflicts, 'prefix'].unique())}")
    return pairs.set_index("prefix")["airline"]


def recover_airline(df: pd.DataFrame, prefix_map: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["airline"] = out["airline"].fillna(flight_prefix(out["flight"]).map(prefix_map))
    return out


def drop_exact_duplicates(train: pd.DataFrame) -> pd.DataFrame:
    """Drop rows identical in every feature AND price (keeps the first copy).
    Left in, a copy can sit in both a training fold and its validation fold, inflating CV scores."""
    cols = [c for c in train.columns if c != ID]
    return train.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def add_flight_key(X: pd.DataFrame) -> pd.DataFrame:
    """Add `flight_key` = flight code + class (e.g. 'UK 819|Business'). The same flight number sells both
    classes at very different prices, so the code alone would mix them. A fixed rule (no prices used)."""
    return X.assign(flight_key=X["flight"] + "|" + X["flight_class"])


def load_clean():
    """Raw data -> (X_train, y_train, X_test, test_ids, sample_submission) with the fixed rules applied."""
    train, test, sample = config.load(TASK)
    prefix_map = build_prefix_map(train, test)
    train = drop_exact_duplicates(recover_airline(train, prefix_map))
    test = recover_airline(test, prefix_map)
    features = [c for c in train.columns if c not in (ID, TARGET)]
    return train[features], train[TARGET], test[features], test[ID], sample


# ---------- learned steps (inside the CV pipeline) ----------

class DurationImputer(TransformerMixin, BaseEstimator):
    """Fill missing `duration` with the median of the same (flight, route, stops) group;
    if that group has no known duration, use the (route, stops) median; then the global median.
    Medians are learned in fit(), so inside CV they come from the training fold only."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        known = X.dropna(subset=["duration"])
        self.group_median_ = known.groupby(DURATION_KEYS)["duration"].median().rename("_group")
        self.fallback_median_ = known.groupby(DURATION_FALLBACK_KEYS)["duration"].median().rename("_fallback")
        self.global_median_ = known["duration"].median()
        return self

    def _lookups(self, X):
        group = X.join(self.group_median_, on=DURATION_KEYS)["_group"]
        fallback = X.join(self.fallback_median_, on=DURATION_FALLBACK_KEYS)["_fallback"]
        return group, fallback

    def transform(self, X):
        group, fallback = self._lookups(X)
        out = X.copy()
        out["duration"] = out["duration"].fillna(group).fillna(fallback).fillna(self.global_median_)
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_  # columns pass through unchanged; needed for set_output("pandas")

    def fill_report(self, X) -> pd.Series:
        """How each missing duration in X would be filled (for the notebook)."""
        group, fallback = self._lookups(X)
        missing = X["duration"].isna()
        by_group = missing & group.notna()
        by_fallback = missing & ~by_group & fallback.notna()
        return pd.Series({"missing": missing.sum(), "group median": by_group.sum(),
                          "route+stops median": by_fallback.sum(),
                          "global median": (missing & ~by_group & ~by_fallback).sum()})


def make_preprocessor(scale: bool = True, rare_airline_min: int = 50, drop_first: bool = False) -> Pipeline:
    """Raw feature DataFrame -> numeric model matrix.

    - duration: imputed (see DurationImputer); duration, days_left: standardised if `scale`
    - stops: ordinal 0/1/2 (it has a natural order)
    - airline: one-hot; airlines with < rare_airline_min training rows share one 'infrequent' column,
      which also absorbs airlines never seen in training
    - other categoricals: one-hot, unseen categories ignored
    - drop_first: drop one column per one-hot group. Plain OLS needs this: with every category kept,
      each group's columns sum to 1 (= the intercept), so coefficients aren't uniquely defined
      (the dummy-variable trap). Ridge/Lasso/trees don't need it.
    - id and the raw flight code are dropped (id is a row number; ~1,070 flight codes are too many)
    """
    def scaler():
        return StandardScaler() if scale else "passthrough"

    drop = "first" if drop_first else None
    columns = ColumnTransformer(
        [
            ("num", scaler(), NUMERIC),
            ("stops", Pipeline([("ord", OrdinalEncoder(categories=[STOPS])), ("scale", scaler())]), ["stops"]),
            ("airline", OneHotEncoder(min_frequency=rare_airline_min, handle_unknown="infrequent_if_exist",
                                      drop=drop, sparse_output=False), ["airline"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("duration", DurationImputer()), ("columns", columns)]).set_output(transform="pandas")


TREE_CATEGORICAL = ["airline", "flight", "flight_key"] + CATEGORICAL


class TreeEncoder(TransformerMixin, BaseEstimator):
    """Raw features -> numeric matrix for tree models (Random Forest, XGBoost).

    - duration imputed exactly as for the linear models (DurationImputer)
    - adds flight_key (flight code × class)
    - stops -> 0/1/2; other categoricals -> one integer code each (learned in fit; unseen -> -1)
    - no scaling and no one-hot: a tree only asks "is this column <= threshold?", so scale is irrelevant,
      and an integer code lets it isolate a category in a few splits instead of needing ~1,500 columns.
    """

    def fit(self, X, y=None):
        Z = add_flight_key(X)
        self.imputer_ = DurationImputer().fit(Z)
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(Z[TREE_CATEGORICAL])
        return self

    def transform(self, X):
        Z = self.imputer_.transform(add_flight_key(X))
        out = pd.DataFrame(self.encoder_.transform(Z[TREE_CATEGORICAL]), columns=TREE_CATEGORICAL, index=Z.index)
        out["stops"] = Z["stops"].map({s: i for i, s in enumerate(STOPS)})
        out[NUMERIC] = Z[NUMERIC]
        return out
