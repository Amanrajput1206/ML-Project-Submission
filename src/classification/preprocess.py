"""Preprocessing for the vehicle-class classification task.

The only fixed rule is deduplication: train rows 7501-10000 are exact copies of earlier rows
(same features, same label). It runs once, before CV, so a row and its copy can never sit in
a training fold and its validation fold at the same time.
Scaling (needed by logistic regression) is learned, so it lives inside each model's Pipeline.
"""
from src import config

TASK = "classification"
TARGET = config.TARGETS[TASK]
ID = config.ID_COLS[TASK]
CLASS_NAMES = {0: "Sedan", 1: "SUV", 2: "Sports Car"}


def load_clean():
    """Raw data -> (X_train, y_train, X_test, test_ids, sample_submission), duplicates dropped."""
    train, test, sample = config.load(TASK)
    features = [c for c in train.columns if c not in (ID, TARGET)]
    train = train.drop_duplicates(subset=features + [TARGET], keep="first").reset_index(drop=True)
    return train[features], train[TARGET], test[features], test[ID], sample
