import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Two-hidden-layer perceptron for binary classification with BatchNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: tuple = (256, 64), dropout: float = 0.4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays in a TensorDataset and return a DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: tuple = (256, 64),
    dropout: float = 0.4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    """Train one MLP fold.

    Returns
    -------
    val_probs : ndarray, shape (N_val,)
    history   : list of {'epoch', 'train_loss', 'val_loss'} per epoch trained
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_weights  = None
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b.to(device)), y_b.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_loader.dataset)

        val_loss, val_probs = _eval(model, val_loader, criterion, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    _, val_probs = _eval(model, val_loader, criterion, device)
    return val_probs, history


def _eval(model, loader, criterion, device) -> tuple[float, np.ndarray]:
    """Run a forward pass over all batches; return mean loss and predicted probabilities."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            total_loss += criterion(logits, y_b.to(device)).item() * len(y_b)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(all_probs)
