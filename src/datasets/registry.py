from src.datasets.base import BaseDataset

_REGISTRY: dict[str, type[BaseDataset]] = {}


def register_dataset(name: str):
    """Return a decorator that registers a dataset loader class under the given name."""
    def decorator(cls: type[BaseDataset]) -> type[BaseDataset]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, path: str, config: dict) -> BaseDataset:
    """Instantiate and return a registered dataset loader by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](path, config)
