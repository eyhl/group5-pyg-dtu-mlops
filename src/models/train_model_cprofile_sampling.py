import cProfile
import logging
import pstats
import sys
from pstats import SortKey

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import NeighborLoader

import wandb
from src.data.make_dataset import load_data
from src.models.model import GCN

sys.path.append("..")

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


# @hydra.main(config_path="../config", config_name="default_config.yaml")
def train():
    # print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    # hparams = config.experiment.hyperparams
    # wandb.config = hparams
    torch.manual_seed(666)
    # orig_cwd = hydra.utils.get_original_cwd()

    # Load data
    data = load_data("data/", name="Cora")
    loader = NeighborLoader(
                            data,
                            # Sample 30 neighbors for each node for 2 iterations
                            num_neighbors=[30] * 2,
                            # Use a batch size of 128 for sampling training nodes
                            batch_size=32,
                            input_nodes=data.train_mask,
                            )
    # sampled_data = next(iter(loader))

    # Model
    model = GCN(
        hidden_channels=16,
        num_features=1433,
        num_classes=7,
        dropout=0.5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 1000

    model = training_loop(epochs, optimizer, criterion, model, loader)

    # Save model
    torch.save(model.state_dict(), "models/" + "checkpoint.pt")

    # Evaluate model
    test_acc = evaluate(model, data)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    wandb.log({"Test accuracy": test_acc})


def training_loop(epochs, optimizer, criterion, model, loader):
    # Train model
    train_loss = []
    for epoch in range(epochs):
        train_loss_batch = []
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
            train_loss_batch.append(loss.item())

        train_loss.append(np.mean(train_loss_batch))
        # print
        print(f"Epoch: {epoch:03d}, Loss: {np.mean(train_loss_batch):.4f}")
        wandb.log({"Training loss": np.mean(train_loss_batch)})
    return model


if __name__ == "__main__":
    cProfile.run('train()', 'restats_batch_sampling')
    p = pstats.Stats('restats_batch_sampling')
    p.sort_stats(SortKey.CUMULATIVE, SortKey.CALLS)
    p.dump_stats('restats_batch_sampling.prof')
    p.print_stats(30)
    # train()
