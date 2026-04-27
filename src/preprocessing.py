import numpy as np
from sklearn.preprocessing import StandardScaler

TEMPORAL_MODES = ('t0', 't1', 'concat', 'delta')


def prepare_features(X: np.ndarray, mode: str) -> np.ndarray:
    """Collapse the time dimension of X into a 2-D feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, 2, 2000)
    mode : one of 't0', 't1', 'concat', 'delta'

    Returns
    -------
    ndarray, shape (N, D)
        D = 2000 for t0/t1/delta; D = 4000 for concat.
    """
    if mode == 't0':
        return X[:, 0, :]
    if mode == 't1':
        return X[:, 1, :]
    if mode == 'concat':
        return np.concatenate([X[:, 0, :], X[:, 1, :]], axis=1)
    if mode == 'delta':
        return X[:, 1, :] - X[:, 0, :]
    raise ValueError(f"mode must be one of {TEMPORAL_MODES}; got '{mode}'")


def fit_transform(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on X_train and return scaled train, scaled val, and the fitted scaler."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s, scaler
