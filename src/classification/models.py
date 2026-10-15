"""Model builders for the classification task that sklearn doesn't provide directly.
Both target the hidden test set's extra measurement noise."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class NoiseAugmented(ClassifierMixin, BaseEstimator):
    """Train `estimator` on the training rows plus `copies` noisy copies of them
    (Gaussian noise, std = k x each feature's std). The model then sees during training the kind of
    blurred inputs it will meet at test time, and learns boundaries that don't depend on fine detail."""

    def __init__(self, estimator, k=0.2, copies=4, random_state=42):
        self.estimator = estimator
        self.k = k
        self.copies = copies
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        sd = X.std().to_numpy()
        noisy = [X + rng.standard_normal(X.shape) * sd * self.k for _ in range(self.copies)]
        self.model_ = clone(self.estimator).fit(pd.concat([X, *noisy]), pd.concat([y] * (self.copies + 1)))
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class NoiseAwareLDA(ClassifierMixin, BaseEstimator):
    """LDA (Gaussian class-conditionals with one shared covariance) for inputs measured with noise.

    If x_noisy = x + e, with e ~ N(0, D) independent of x and D = diag((k * std_j)^2), then within each
    class x_noisy ~ N(mu_c, Sigma + D). So the Bayes classifier for *noisy* inputs is ordinary LDA with
    Sigma replaced by Sigma + D. k = 0 gives plain LDA.
    """

    def __init__(self, k=0.2):
        self.k = k

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.priors_ = np.array([(y == c).mean() for c in self.classes_])
        centred = X - self.means_[np.searchsorted(self.classes_, y)]
        cov = np.cov(centred.T) + np.diag((self.k * X.std(axis=0)) ** 2)
        precision = np.linalg.inv(cov)
        # Linear discriminant per class: x' P mu_c - mu_c' P mu_c / 2 + log prior_c
        self.coef_ = self.means_ @ precision
        self.intercept_ = -0.5 * np.sum(self.coef_ * self.means_, axis=1) + np.log(self.priors_)
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def predict_proba(self, X):
        d = self.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
