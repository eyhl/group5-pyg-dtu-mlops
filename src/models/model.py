import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv  # type: ignore


class GCN(nn.Module):
    def __init__(
        self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "Expected input is not a 2D tensor," f"instead it is a {x.ndim}D tensor."
            )
        if x.shape[1] != 1433:
            raise ValueError("Feature vector should be of size 1433.")
        x = self.conv1(x, edge_index)
        x = self.dropout(F.relu(x))
        x = self.conv2(x, edge_index)
        return x


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = GCN(checkpoint["hidden_channels"],
                checkpoint["num_features"],
                checkpoint["num_classes"],
                checkpoint["dropout"])
    model.load_state_dict(checkpoint['state_dict'])

    return model
