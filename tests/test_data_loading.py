import numpy as np
import pytest
from src.data import load_task, task_summary

H5 = 'data/temporal_two_sets_n2000.h5'


@pytest.mark.parametrize('task,expected_n', [('task1', 190), ('task2', 301)])
def test_load_shape(task, expected_n):
    X, y, cpg_ids = load_task(H5, task)
    assert X.shape == (expected_n, 2, 2000)
    assert y.shape == (expected_n,)
    assert cpg_ids.shape == (2000,)


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_beta_value_range(task):
    X, _, _ = load_task(H5, task)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_no_nans(task):
    X, y, _ = load_task(H5, task)
    assert not np.isnan(X).any()
    assert not np.isnan(y.astype(float)).any()


@pytest.mark.parametrize('task,expected_pos', [('task1', 43), ('task2', 110)])
def test_label_counts(task, expected_pos):
    _, y, _ = load_task(H5, task)
    assert y.sum() == expected_pos


def test_invalid_task():
    with pytest.raises(ValueError):
        load_task(H5, 'task99')


@pytest.mark.parametrize('task', ['task1', 'task2'])
def test_task_summary(task):
    s = task_summary(H5, task)
    assert s['n_positive'] + s['n_negative'] == s['n_total']
    assert 0 < s['positive_rate'] < 1
