from abc import ABC, abstractmethod
import numpy as np


class BaseDataset(ABC):
    """Common interface every dataset loader must implement.

    load_task returns:
        X          : ndarray (N, T, D) — individuals × timepoints × features
        y          : ndarray (N,)      — 0 = non-converter, 1 = converter
        feature_ids: ndarray (D,)      — feature identifiers (e.g. CpG names)
    """

    @abstractmethod
    def load_task(self, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, feature_ids) for the named task."""
        ...

    @abstractmethod
    def available_tasks(self) -> list[str]:
        """Return the list of task names this dataset exposes."""
        ...

    @abstractmethod
    def task_description(self, task: str) -> str:
        """Return a human-readable description of the named task."""
        ...
