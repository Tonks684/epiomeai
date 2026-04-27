import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def get_cv(n_splits: int = 5, n_repeats: int = 3, random_state: int = 42) -> RepeatedStratifiedKFold:
    """Return a configured RepeatedStratifiedKFold cross-validator."""
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def assert_no_leakage(train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Raise AssertionError if any index appears in both train and validation sets."""
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    assert train_set.isdisjoint(val_set), (
        f"Leakage detected: {len(train_set & val_set)} indices appear in both train and val"
    )
