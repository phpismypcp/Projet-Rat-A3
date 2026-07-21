"""Train the Transformer auto-encoder on normal transactions only.

The model learns to reconstruct normal transactions. We monitor reconstruction
loss on the *normal rows of the validation set* (unseen normal data) for early
stopping, so training halts once it stops generalizing.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomaly_explainer.config import MODEL, TRAIN, ModelConfig, TrainConfig
from anomaly_explainer.data.preprocess import PreparedData
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder

History = dict[str, list[float]]


def resolve_device() -> torch.device:
    """Use CUDA when available (fast on a GPU machine), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # attention kernels lack a det path


def _normal_val_tensor(prepared: PreparedData) -> torch.Tensor:
    """Validation features for label==0 rows (generalization on normal data)."""
    mask = prepared.y_val == 0
    return torch.from_numpy(prepared.X_val[mask])


def _epoch_loss(model: nn.Module, x: torch.Tensor, loss_fn: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(x), x).item())


def train_autoencoder(
    prepared: PreparedData,
    model_config: ModelConfig = MODEL,
    train_config: TrainConfig = TRAIN,
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
) -> tuple[nn.Module, History]:
    """Train on ``prepared.X_train`` (normal-only) with early stopping.

    Args:
        on_epoch_end: optional callback ``(epoch, train_loss, val_loss)`` invoked
            after each epoch, e.g. for live progress logging.
        model_factory: builds the network to train; defaults to the Transformer
            auto-encoder. Supplying a factory lets the dense baseline reuse this
            exact loop — same seed, batching, optimizer and early stopping — so a
            comparison between architectures isn't confounded by the setup.

    Returns the model (restored to its best-validation weights) and a history of
    per-epoch train/val reconstruction losses.
    """
    _seed_everything(train_config.random_seed)
    device = resolve_device()

    build = model_factory or (lambda: TransformerAutoEncoder(model_config))
    model = build().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    x_train = torch.from_numpy(prepared.X_train)
    loader = DataLoader(
        TensorDataset(x_train),
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(train_config.random_seed),
    )
    x_val = _normal_val_tensor(prepared).to(device)

    history: History = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = model.state_dict()
    epochs_without_improvement = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch.shape[0]

        train_loss = running / len(x_train)
        val_loss = _epoch_loss(model, x_val, loss_fn) if len(x_val) else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return model, history
