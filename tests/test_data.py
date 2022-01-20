import torch
from torch_geometric.datasets import Planetoid  # type: ignore
from torch_geometric.transforms import NormalizeFeatures  # type: ignore


def test_data_loading_output_shape():
    dataset = Planetoid(root="data/", name="Cora", transform=NormalizeFeatures())
    assert len(dataset[0]) == 6, "Data was not probably loaded - not all dimensions exists"


def test_data_loading_output_is_tensor():
    dataset = Planetoid(root="data/", name="Cora", transform=NormalizeFeatures())
    assert torch.is_tensor(dataset[0].x), "Nodes are not tensor"
    assert torch.is_tensor(dataset[0].edge_index), "Edges are not tensor"
    assert torch.is_tensor(dataset[0].y), "Classes is not tensor"
    assert torch.is_tensor(dataset[0].train_mask), "Train masks are not tensor"
    assert torch.is_tensor(dataset[0].val_mask), "Val masks are not tensor"
    assert torch.is_tensor(dataset[0].test_mask), "Test masks are not tensor"


def test_data_loading_output_element_shape():
    dataset = Planetoid(root="data/", name="Cora", transform=NormalizeFeatures())
    torch_size = torch.ones((1, 1433))
    assert [
        a.shape == torch_size for a in dataset[0].x
    ], "Data tensors do not have the correct shape"


def test_data_loading_classes():
    dataset = Planetoid(root="data/", name="Cora", transform=NormalizeFeatures())
    unique_classes = torch.unique(dataset[0].y)
    torch_unique = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    assert torch.equal(unique_classes, torch_unique), "Not all classes are represented in the data"


if __name__ == "__main__":
    test_data_loading_output_shape()
    test_data_loading_output_is_tensor()
    test_data_loading_output_element_shape()
    test_data_loading_classes()
