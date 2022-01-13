import logging
import sys

sys.path.append("..")

import wandb
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

from src.data.make_dataset import load_data
from src.models.model import GCN

log = logging.getLogger(__name__)
print = log.info
wandb.init(project="group5-pyg-dtumlops", entity="group5-dtumlops")

def evaluate(model: nn.Module, data: torch_geometric.data.Data) -> float:
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    wandb.config = hparams
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(hparams["seed"])
    orig_cwd = hydra.utils.get_original_cwd()

    # Load data
    data = load_data(orig_cwd + "/data/", name="Cora")
    loader = DataLoader(data, batch_size=32, shuffle=True)
    # Model
    model = GCN(
        hidden_channels=hparams["hidden_channels"],
        num_features=hparams["num_features"],
        num_classes=hparams["num_classes"],
        dropout=hparams["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = hparams["epochs"]
    train_loss = []

    # Train model
    for epoch in range(epochs):
        for batch in loader:
        # Clear gradients
            optimizer.zero_grad()
            # Perform a single forward pass
            out = model(batch.x, batch.edge_index)
            # Compute the loss solely based on the training nodes
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            # Derive gradients
            loss.backward()
            # Update parameters based on gradients
            optimizer.step()
            # Append results
            train_loss.append(loss.item())
            # print
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            wandb.log({"Training loss": loss})

    # Save model
    torch.save(model.state_dict(), orig_cwd + "/models/" + hparams["checkpoint_name"])

    # Evaluate model
    test_acc = evaluate(model, data[0])
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    wandb.log({"Test accuracy": test_acc})


if __name__ == "__main__":
    train()
