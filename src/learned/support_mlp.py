import torch
import torch.nn as nn


class SupportMLP(nn.Module):
    """Small coordinate-wise MLP for support probability prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


@torch.no_grad()
def predict_topk_support(model, features, k: int, device: str = "cpu"):
    """Predict top-k support indices using the learned model."""
    model.eval()
    x = torch.tensor(features, dtype=torch.float32, device=device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    support = torch.topk(probs, k=k).indices.cpu().numpy()
    return support
