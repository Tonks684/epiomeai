import numpy as np
import pytest
from src.data import load_task, task_summary

H5 = 'data/temporal_two_sets_n2000.h5'


@pytest.mark.parametrize('task,expected_n', [('task1', 190), ('task2', 301)])
def test_load_shape(task, expected_n):
    """X is (N, 2, 2000), y is (N,), and cpg_ids is (2000,) for each task."""
    X, y, cpg_ids = load_task(H5, task)
    assert X.shape == (expected_n, 2, 2000)
    assert y.shape == (expected_n,)
    assert cpg_ids.shape == (2000,)


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_beta_value_range(task):
    """All methylation beta values lie within the valid [0, 1] range."""
    X, _, _ = load_task(H5, task)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_no_nans(task):
    """Feature matrix and labels contain no NaN values."""
    X, y, _ = load_task(H5, task)
    assert not np.isnan(X).any()
    assert not np.isnan(y.astype(float)).any()


@pytest.mark.parametrize('task,expected_pos', [('task1', 43), ('task2', 110)])
def test_label_counts(task, expected_pos):
    """Positive label counts match the known converter counts from the HDF5 groups."""
    _, y, _ = load_task(H5, task)
    assert y.sum() == expected_pos


def test_invalid_task():
    """Requesting a task name not in the config raises a ValueError."""
    with pytest.raises(ValueError):
        load_task(H5, 'task99')


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_task_summary(task):
    """task_summary returns consistent counts and a positive rate strictly between 0 and 1."""
    s = task_summary(H5, task)
    assert s['n_positive'] + s['n_negative'] == s['n_total']
    assert 0 < s['positive_rate'] < 1
