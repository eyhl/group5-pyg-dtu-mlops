import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf

import sys
sys.path.append("..")
from src.data.make_dataset import load_data
from src.models.model import GCN


@hydra.main(config_path="../config", config_name='default_config.yaml')
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(hparams["seed"])

    # Load data
    data = load_data("../../data/", name = "Cora")

    # Model 
    model = GCN(hidden_channels=hparams["hidden_channels"], num_features=hparams["num_features"], num_classes=hparams["num_classes"], dropout=hparams["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = hparams["epochs"]
    train_loss = []
    train_accuracy = []

    # Train model
    for epoch in range(epochs):
        # Clear gradients
        optimizer.zero_grad()  
        # Perform a single forward pass
        out = model(data.x, data.edge_index)  
        # Compute the loss solely based on the training nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
        # Derive gradients 
        loss.backward()  
        # Update parameters based on gradients
        optimizer.step() 
        # Append results
        train_loss.append(loss.item())
        # print 
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
     


if __name__ == "__main__":
    train()