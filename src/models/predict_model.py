import hydra
import torch
from omegaconf import OmegaConf

from src.data.make_dataset import load_data
from src.models.model import load_checkpoint


@hydra.main(config_path="../config", config_name="default_config.yaml")
def predict(config) -> None:
    """
    Computes accuracy on validation set.
    :param config: Config file used for Hydra
    :return:
    """
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    torch.manual_seed(hparams["seed"])
    orig_cwd = hydra.utils.get_original_cwd()

    # Load data
    data = load_data(orig_cwd + "/data/", name="Cora")

    # Model
    path_to_model = orig_cwd + hparams.load_model_from + hparams.checkpoint_name
    model = load_checkpoint(path_to_model)

    # Make prediction
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    valid_correct = pred[data.val_mask] == data.y[data.val_mask]
    # Derive ratio of correct predictions.
    valid_acc = int(valid_correct.sum()) / int(data.val_mask.sum())

    print(f"Prediction accuracy: {valid_acc * 100:.2f}%")


if __name__ == "__main__":
    predict()
