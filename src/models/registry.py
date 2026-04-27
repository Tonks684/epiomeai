from typing import Callable, Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    """Return a decorator that registers a model builder function under the given name."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_model(name: str, config: dict):
    """Instantiate and return a registered model by name, passing config to its builder."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](config)


def registered_models() -> list[str]:
    """Return a list of all registered model names."""
    return list(_REGISTRY.keys())
