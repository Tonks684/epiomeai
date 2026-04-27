"""Thin compatibility shim — delegates to src.datasets.AdniDataset.

Training scripts and tests that call load_task() / task_summary() directly
continue to work unchanged. New code should use src.datasets.get_dataset()
so that the dataset is driven entirely by config.
"""
import numpy as np
from src.datasets.adni import AdniDataset, _DEFAULT_TASKS

# Kept for backward compatibility and inspect_hdf5.py
TASK_GROUPS = {
    task: {
        'negative':    cfg['negative'],
        'positive':    cfg['positive'],
        'cpg_ids':     cfg['feature_ids'],
        'label_names': tuple(cfg['description'].split(' vs ')),
    }
    for task, cfg in _DEFAULT_TASKS.items()
}

_DEFAULT_H5 = None
_dataset_cache: dict[str, AdniDataset] = {}


def _get(h5_path: str) -> AdniDataset:
    """Return a cached AdniDataset for the given HDF5 path."""
    if h5_path not in _dataset_cache:
        _dataset_cache[h5_path] = AdniDataset(h5_path, {})
    return _dataset_cache[h5_path]


def load_task(h5_path: str, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, y, feature_ids) for a named task from the given HDF5 file."""
    return _get(h5_path).load_task(task)


def task_summary(h5_path: str, task: str) -> dict:
    """Return a dict of count and rate summary statistics for a task."""
    return _get(h5_path).task_summary(task)
