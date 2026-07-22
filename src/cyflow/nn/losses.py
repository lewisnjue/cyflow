from cyflow import Tensor


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean Squared Error Loss."""
    return ((predictions - targets) ** 2).mean()


def rmse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Root Mean Squared Error Loss."""
    return (((predictions - targets) ** 2).mean()).sqrt()


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross Entropy Loss for multi-class classification."""
    log_probs = logits.log_softmax(axis=-1)
    ce_loss = -(targets * log_probs).sum(axis=-1).mean()
    return ce_loss


def binary_cross_entropy_loss(predictions: Tensor, targets: Tensor, eps: float = 1e-7) -> Tensor:
    """Binary Cross Entropy Loss for binary classification."""
    predictions = predictions.clip(eps, 1.0 - eps)
    bce_loss = -(
        targets * predictions.log() + (1 - targets) * (1 - predictions).log()
    ).mean()
    return bce_loss


def logits_binary_cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary Cross Entropy Loss taking raw logits as input."""
    probs = logits.sigmoid()
    return binary_cross_entropy_loss(probs, targets)
