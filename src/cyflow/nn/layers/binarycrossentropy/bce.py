import cyflow
import numy as np


class BinaryCrossEntropy:
    def __init__(self, device="cpu"):
        assert device in ["cpu", "cuda"] or isinstance(device, int), "Device must be either 'cpu' or 'cuda' or an integer constant"
        self.device = device

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
        # Clip predictions to avoid log(0)
        """
        eps = 1e-12
        predictions_clipped = cyflow.clip(predictions, eps, 1 - eps)
        loss = - (targets * cyflow.log(predictions_clipped) + (1 - targets) * cyflow.log(1 - predictions_clipped))
        return loss.mean()
        """ # most methods not imremented yet i will just return a fancy loss for now 
        loss = cyflow.tensor(data=np.random.randn(1).tolist(), shape=(1,), device=self.device)