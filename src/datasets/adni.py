import h5py
import numpy as np

from src.datasets.base import BaseDataset
from src.datasets.registry import register_dataset


@register_dataset('adni')
class AdniDataset(BaseDataset):
    """Loader for the ADNI HDF5 methylation file (temporal_two_sets_n2000.h5).

    Task definitions are driven by the 'tasks' block in the dataset config:

        tasks:
          task1:
            negative: X_cn_to_cn
            positive: X_cn_to_mci
            feature_ids: cpg_ids_cn
            description: "CN stays CN vs CN converts to MCI"
          task2:
            ...

    X shape in the file: (features, timepoints, individuals).
    Returns X transposed to (individuals, timepoints, features).
    """

    def __init__(self, path: str, config: dict):
        self.path   = path
        self._tasks = config.get('tasks', _DEFAULT_TASKS)

    def available_tasks(self) -> list[str]:
        """Return the list of configured task names."""
        return list(self._tasks.keys())

    def task_description(self, task: str) -> str:
        """Return the description string for the named task."""
        return self._tasks[task].get('description', task)

    def load_task(self, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load HDF5 groups for the task, transpose to (N, T, D), and assign binary labels."""
        if task not in self._tasks:
            raise ValueError(f"Task '{task}' not in config. Available: {self.available_tasks()}")

        cfg = self._tasks[task]
        with h5py.File(self.path, 'r') as f:
            X_neg = f[cfg['negative']][:].transpose(2, 1, 0).astype(np.float32)
            X_pos = f[cfg['positive']][:].transpose(2, 1, 0).astype(np.float32)
            feature_ids = f[cfg['feature_ids']][:].astype(str)

        X = np.concatenate([X_neg, X_pos], axis=0)
        y = np.concatenate([
            np.zeros(len(X_neg), dtype=np.int64),
            np.ones(len(X_pos),  dtype=np.int64),
        ])
        return X, y, feature_ids

    def task_summary(self, task: str) -> dict:
        """Return a dict of individual counts, positive rate, and shape metadata for a task."""
        X, y, _ = self.load_task(task)
        return {
            'task':          task,
            'n_negative':    int((y == 0).sum()),
            'n_positive':    int((y == 1).sum()),
            'n_total':       len(y),
            'positive_rate': float(y.mean()),
            'n_timepoints':  X.shape[1],
            'n_features':    X.shape[2],
        }


# Fallback if no 'tasks' block is provided in config
_DEFAULT_TASKS = {
    'task1': {
        'negative':    'X_cn_to_cn',
        'positive':    'X_cn_to_mci',
        'feature_ids': 'cpg_ids_cn',
        'description': 'CN stays CN vs CN converts to MCI',
    },
    'task2': {
        'negative':    'X_mci_to_mci',
        'positive':    'X_mci_to_dem',
        'feature_ids': 'cpg_ids_mci',
        'description': 'MCI stays MCI vs MCI converts to dementia',
    },
}
