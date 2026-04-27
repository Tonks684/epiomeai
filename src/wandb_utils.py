"""Thin W&B wrapper — silently no-ops when W&B is disabled or not installed."""
try:
    import wandb as _wandb
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class WandbLogger:
    """Thin W&B wrapper that silently no-ops when W&B is disabled or not installed."""

    def __init__(self, enabled: bool, project: str, name: str, config: dict):
        """Initialise the logger and start a W&B run if enabled and W&B is available."""
        self._active = enabled and _AVAILABLE
        if self._active:
            _wandb.init(project=project, name=name, config=config, reinit=True)
        elif enabled and not _AVAILABLE:
            print("Warning: --wandb requested but 'wandb' is not installed. Skipping.")

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log a dict of scalar metrics at the given step."""
        if self._active:
            _wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict) -> None:
        """Write scalar metrics to the W&B run summary."""
        if self._active:
            for k, v in metrics.items():
                _wandb.run.summary[k] = v

    def finish(self) -> None:
        """Finish the active W&B run if one is running."""
        if self._active:
            _wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()
