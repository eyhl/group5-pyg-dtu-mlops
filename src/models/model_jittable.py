import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore


class GCN(nn.Module):
    def __init__(
        self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels).jittable()
        self.conv2 = GCNConv(hidden_channels, num_classes).jittable()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.copy()
        edge_index = data.edge_index.copy()
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
