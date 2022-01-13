import pytest
import torch

from src.models.model import GCN


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


def test_on_wrong_number_of_x_y():
    model = GCN(
        hidden_channels=16,
        num_features=1433,
        num_classes=7,
        dropout=0.5,
    )
    with pytest.raises(ValueError, match="Number of training examples."):
        model(torch.randn(2, 1433), torch.randn(3))  # 2 != 3

if __name__ == "__main__":
    test_on_wrong_dimension_to_forward()
    test_on_wrong_feature_dimension_to_forward()
    test_on_wrong_number_of_x_y()