import hydra
from omegaconf import OmegaConf

import torch

from src.data.make_dataset import load_data
from src.models.model import GCN


@hydra.main(config_path="../config", config_name='default_config.yaml')
def predict(config) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(hparams["seed"])

    # Load data
    data = load_data("../../data/", name = "Cora")

    # Model 
    model = GCN(hidden_channels=hparams["hidden_channels"], num_features=hparams["num_features"], num_classes=hparams["num_classes"], dropout=hparams["dropout"])
    # Load parameters
    state_dict = torch.load(hparams.load_model_from)
    model.load_state_dict(state_dict)

    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    valid_correct = pred[data.val_mask] == data.y[data.val_mask]
    # Derive ratio of correct predictions.
    valid_acc = int(valid_correct.sum()) / int(data.val_mask.sum())

    print(f'Prediction accuracy: {valid_acc * 100:.2f}%')

if __name__ == "__main__":
    train()