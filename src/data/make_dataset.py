# -*- coding: utf-8 -*-
import torch_geometric.data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_data(path: str, name: str) -> torch_geometric.data.Data:
    """
    Loads the data from {path}/{name} or downloads it and saves it into the directory.
    :param path: Path to the dataset
    :param name: Name of the dataset
    :return: data -- the inputs are stored in data.x, labels in data.y,
                     train and test masks in data.train_mask and data.test_mask
    """
    # Get torch_geometric.data.InMemoryDataset dataset
    dataset = Planetoid(root=path, name=name, transform=NormalizeFeatures())
    # Return only the data
    return dataset
