"""Model builders for the regression task. Every model takes the raw feature DataFrame
(airline already recovered), so preprocessing is part of the model and re-fits per CV fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone


class GroupMedianRegressor(RegressorMixin, BaseEstimator):
    """Baseline: predict the training median price of the row's group (e.g. its flight_class).
    With group_col=None it predicts the single overall median."""

    def __init__(self, group_col=None):
        self.group_col = group_col

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_median_ = float(y.median())
        self.medians_ = {} if self.group_col is None else y.groupby(X[self.group_col]).median().to_dict()
        return self

    def predict(self, X):
        if self.group_col is None:
            return np.full(len(X), self.global_median_)
        return X[self.group_col].map(self.medians_).fillna(self.global_median_).to_numpy(dtype=float)


class PerGroupRegressor(RegressorMixin, BaseEstimator):
    """Fit a separate copy of `estimator` for each value of `group_col` (e.g. one model for Economy,
    one for Business) and route each row to its group's model at prediction time."""

    def __init__(self, estimator, group_col="flight_class"):
        self.estimator = estimator
        self.group_col = group_col

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.models_ = {g: clone(self.estimator).fit(X[m], y[m])
                        for g, m in ((g, X[self.group_col] == g) for g in X[self.group_col].unique())}
        return self

    def predict(self, X):
        pred = np.empty(len(X))
        for g, model in self.models_.items():
            m = (X[self.group_col] == g).to_numpy()
            if m.any():
                pred[m] = model.predict(X[m])
        return pred


class ClassInteractions(TransformerMixin, BaseEstimator):
    """Append `column × class-indicator` for every encoded column, so a linear model can learn a
    different coefficient per class for each feature (e.g. 'premium airline' adds more in Business).
    Works on the encoded DataFrame from make_preprocessor; the indicator is the first flight_class_* column."""

    def fit(self, X, y=None):
        cls_cols = sorted(c for c in X.columns if c.startswith("flight_class_"))
        self.indicator_ = cls_cols[0]
        self.base_cols_ = [c for c in X.columns if not c.startswith("flight_class_")]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        inter = X[self.base_cols_].mul(X[self.indicator_], axis=0)
        inter.columns = [f"{c} x {self.indicator_}" for c in self.base_cols_]
        return pd.concat([X, inter], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + [f"{c} x {self.indicator_}" for c in self.base_cols_],
                          dtype=object)


class FloorRegressor(RegressorMixin, BaseEstimator):
    """Wrap a regressor and raise any prediction below the cheapest training price up to that price.
    A linear model can extrapolate below zero for cheap Economy rows; no real ticket costs that."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.model_ = clone(self.estimator).fit(X, y)
        self.floor_ = float(np.min(y))
        return self

    def predict(self, X):
        return np.maximum(self.model_.predict(X), self.floor_)
