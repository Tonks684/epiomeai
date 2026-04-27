from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.registry import register_model


@register_model('logreg')
def build_logreg(config: dict) -> LogisticRegression:
    """Build an L2 logistic regression classifier with balanced class weights."""
    cfg = config.get('logreg', {})
    return LogisticRegression(
        C=cfg.get('C', 1.0),
        penalty=cfg.get('penalty', 'l2'),
        class_weight='balanced',
        solver='lbfgs',
        max_iter=cfg.get('max_iter', 1000),
        random_state=42,
    )


@register_model('gbm')
def build_gbm(config: dict) -> HistGradientBoostingClassifier:
    """Build a histogram gradient-boosted trees classifier with balanced class weights."""
    cfg = config.get('gbm', {})
    return HistGradientBoostingClassifier(
        max_iter=cfg.get('max_iter', 200),
        learning_rate=cfg.get('learning_rate', 0.05),
        max_depth=cfg.get('max_depth', 4),
        min_samples_leaf=cfg.get('min_samples_leaf', 20),
        class_weight='balanced',
        random_state=42,
    )


def build_model(config: dict):
    """Delegate to the model registry to instantiate the model named in config['model']."""
    from src.models.registry import get_model
    return get_model(config['model'], config)
