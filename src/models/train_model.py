import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
import torch_geometric
import logging

import sys
sys.path.append("..")
from src.data.make_dataset import load_data
from src.models.model import GCN

log = logging.getLogger(__name__)
print = log.info

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

@hydra.main(config_path="../config", config_name='default_config.yaml')
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(hparams["seed"])
    orig_cwd = hydra.utils.get_original_cwd()

    # Load data
    data = load_data(orig_cwd + "/data/", name = "Cora")

    # Model 
    model = GCN(hidden_channels=hparams["hidden_channels"], num_features=hparams["num_features"], num_classes=hparams["num_classes"], dropout=hparams["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = hparams["epochs"]
    train_loss = []

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

    # Save model
    torch.save(model.state_dict(), orig_cwd + '/models/' + hparams["checkpoint_name"])

    # Evaluate model
    test_acc = evaluate(model, data)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

if __name__ == "__main__":
    train()