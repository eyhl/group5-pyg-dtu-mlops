import pytest
import torch

from src.data.make_dataset import load_data
from src.models.model import GCN


def test_model_input_output():
    data = load_data("data/", name="Cora")

    N_nodes = data.x.shape[0]
    N_features = 1433
    N_classes = 7

    model = GCN(
        hidden_channels=16,
        num_features=1433,
        num_classes=N_classes,
        dropout=0.5,
    )

    # Check for the input dimension
    assert data.x.shape == torch.Size(
        [N_nodes, N_features]
    ), f"Incorrect shape of node features. Feature vector should be {N_features} elements long"

    assert data.y.shape == torch.Size(
        [N_nodes]
    ), "Number of targets ({data.x.shape}) is not equal to the number of nodes ({data.y.shape})."
    out = model(data.x, data.edge_index)

    # Check for the output dimension
    assert out.shape == torch.Size([N_nodes, N_classes]), "Incorrect shape of model output."


def test_on_wrong_dimension_to_forward():
    model = GCN(
        hidden_channels=16,
        num_features=1433,
        num_classes=7,
        dropout=0.5,
    )
    with pytest.raises(ValueError, match="Expected input is not a 2D tensor."):
        model(torch.randn(1, 1433, 2), torch.randn(1))


def test_on_wrong_feature_dimension_to_forward():
    model = GCN(
        hidden_channels=16,
        num_features=1433,
        num_classes=7,
        dropout=0.5,
    )
    with pytest.raises(ValueError, match="Feature vector should be of size 1433."):
        model(torch.randn(1, 143), torch.randn(1))


if __name__ == "__main__":
    test_model_input_output()
    test_on_wrong_dimension_to_forward()
    test_on_wrong_feature_dimension_to_forward()
