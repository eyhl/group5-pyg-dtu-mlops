import cProfile
import logging
import pstats
import sys
from pstats import SortKey

import numpy as np
import torch
import torch.nn as nn
import torch_geometric  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore

import wandb
from src.data.make_dataset import load_data
from src.models.model import GCN

sys.path.append("..")

log = logging.getLogger(__name__)
print = log.info
wandb.init(project="group5-pyg-dtumlops", entity="group5-dtumlops")


def evaluate(model: nn.Module, data: torch_geometric.data.Data) -> float:
    """
    Evaluates model on data and returns accuracy.
    :param model: Model to be evaluated
    :param data: Data to evaluate on
    :return: accuracy
    """
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


def train():
    """
    Trains the model with manual hyperparameters on train data,
    saves the model and evaluates it on test data.
    :return:
    """
    torch.manual_seed(666)

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


def training_loop(
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
    model: nn.Module,
    loader: torch_geometric.loader,
) -> nn.Module:
    """
    Training loop
    :return: model
    """
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
    """
    Run cProling, save in a .prof file, and print top 30
    :return:
    """
    cProfile.run("train()", "reports/restats_batch_sampling")
    p = pstats.Stats("reports/restats_batch_sampling")
    p.sort_stats(SortKey.CUMULATIVE, SortKey.CALLS)
    p.dump_stats("reports/restats_batch_sampling.prof")
    p.print_stats(30)
