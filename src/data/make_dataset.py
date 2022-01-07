# -*- coding: utf-8 -*-
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

s
def load_data(path, name):
    dataset = Planetoid(root=path, name=name, transform=NormalizeFeatures())
    return dataset[0]
