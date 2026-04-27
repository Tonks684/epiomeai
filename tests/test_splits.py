import numpy as np
import pytest
from src.data import load_task
from src.preprocessing import prepare_features
from src.splits import assert_no_leakage, get_cv

H5 = 'data/temporal_two_sets_n2000.h5'


def test_no_individual_leakage():
    X_raw, y, _ = load_task(H5, 'task1')
    X = prepare_features(X_raw, 't0')
    cv = get_cv(n_splits=5, n_repeats=1)
    for train_idx, val_idx in cv.split(X, y):
        assert_no_leakage(train_idx, val_idx)


def test_cv_covers_all_individuals():
    X_raw, y, _ = load_task(H5, 'task1')
    X = prepare_features(X_raw, 't0')
    cv = get_cv(n_splits=5, n_repeats=1)
    seen = set()
    for _, val_idx in cv.split(X, y):
        seen.update(val_idx.tolist())
    assert seen == set(range(len(y)))


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_stratification_preserves_ratio(task):
    X_raw, y, _ = load_task(H5, task)
    X = prepare_features(X_raw, 't0')
    overall_rate = y.mean()
    cv = get_cv(n_splits=5, n_repeats=1)
    for train_idx, val_idx in cv.split(X, y):
        val_rate = y[val_idx].mean()
        assert abs(val_rate - overall_rate) < 0.15, (
            f"Fold positive rate {val_rate:.2f} deviates too far from {overall_rate:.2f}"
        )
