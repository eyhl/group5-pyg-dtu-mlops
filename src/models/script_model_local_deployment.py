from omegaconf import OmegaConf
import torch
import hydra
from src.models.model_jittable import GCN
from src.data.make_dataset import load_data
import pickle


@hydra.main(config_path="../config", config_name="default_config.yaml")
def export_scripted_model(config) -> None:

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment.hyperparams
    orig_cwd = hydra.utils.get_original_cwd()

    model = GCN(
        hidden_channels=hparams["hidden_channels"],
        num_features=hparams["num_features"],
        num_classes=hparams["num_classes"],
        dropout=hparams["dropout"],
    )

    path_to_model = orig_cwd + hparams.load_model_from + hparams.checkpoint_name
    # state_dict = torch.load(path_to_model)
    state_dict = torch.load(orig_cwd + "/models/checkpoint_jittable.pt")
    model.load_state_dict(state_dict)

    data = load_data(orig_cwd + "/data/", name="Cora")

    script_model = torch.jit.script(model, example_inputs=[hparams["hidden_channels"],
                                                           hparams["num_features"],
                                                           hparams["num_classes"],
                                                           hparams["dropout"]])

    out_unscripted = model(data.x, data.edge_index)
    pred_unscripted = out_unscripted.argmax(dim=1)
    unscripted_top5_indices = pred_unscripted.topk(5).indices

    out_scripted = script_model(data.x, data.edge_index)
    pred_scripted = out_scripted.argmax(dim=1)
    scripted_top5_indices = pred_scripted.topk(5).indices
    print("Unscripted:", unscripted_top5_indices, "Scripted:", scripted_top5_indices)

    if torch.allclose(unscripted_top5_indices, scripted_top5_indices):
        # Save model
        # directory = orig_cwd + "/models/"
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filename = directory + "deployable_model.pt"
        # torch.save(script_model, orig_cwd + "/models/deployable_model.pt")
        # with open(orig_cwd + "/models/deployable_model.pt", 'wb') as handle:
            # pickle.dump(script_model, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    export_scripted_model()
