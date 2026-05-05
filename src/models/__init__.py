from src.models.registry import get_model, register_model
from src.models import sklearn_models, mlp  # noqa: F401 — trigger @register_model decorators

__all__ = ['get_model', 'register_model']
