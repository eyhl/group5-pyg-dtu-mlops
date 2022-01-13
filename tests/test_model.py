import pytest
import torch

from src.models.model import GCN
from src.data.make_dataset import load_data
from torch_geometric.loader import DataLoader


def test_model_input_output():
    batch_size = 32
    data = load_data("data/", name="Cora")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    model = GCN(
        hidden_channels=16,
        num_features=1433,
        num_classes=7,
        dropout=0.5,
    )

    last_batch_size = len(data[0].x) % batch_size

    for step, batch in enumerate(loader):
        # if this is not the last batch
        if step != len(loader) - 1:
            # check for the dimension of input
            assert batch.x[batch.train_mask].shape == torch.Size([batch_size, 1433])
            out = model(batch.x[batch.train_mask], batch.edge_index[batch.train_mask])
            # check for the dimension of output
            assert out.shape == torch.Size([batch_size, 10])
        else:
            # check for the dimension of input
            assert batch.x[batch.train_mask].shape == torch.Size([last_batch_size, 1433])
            out = model(batch.x[batch.train_mask], batch.edge_index[batch.train_mask])
            # check for the dimension of output
            assert out.shape == torch.Size([last_batch_size, 10])


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
    # test_on_wrong_dimension_to_forward()
    # test_on_wrong_feature_dimension_to_forward()
    # test_on_wrong_number_of_x_y()
    test_model_input_output()