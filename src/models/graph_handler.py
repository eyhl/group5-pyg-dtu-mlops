import torch
import os
# from ts.torch_handler.base_handler import BaseHandler
import yaml
from torch_geometric.nn import GCNConv


class CustomGraphHandler():

    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self):
        model_path = "models/deployable_model.pt"
        model_config_path = "src/config/exp1.yaml"
        with open(model_config_path) as file:
            hparams = yaml.load(file, Loader=yaml.FullLoader)

        model = GCNConv(hidden_channels=hparams["hidden_channels"],
                        num_features=hparams["num_features"],
                        num_classes=hparams["num_classes"],
                        dropout=hparams["dropout"])

        model.load_state_dict(torch.load(model_path))
        self.model = model.eval()
        print("Loaded model:", model)

        self.initialize = True

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        data = data['data']
        x = data.x.clone()
        edge_index = data.edge_index.clone()
        return (
            torch.as_tensor(x, device=self.device),
            torch.as_tensor(edge_index, device=self.device),
        )

    def inference(self, x, edge_index):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(x, edge_index)
        return model_output

    def handle(self, data):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """

        # Get path to data from extra files
        # properties = context.system_properties
        # model_dir = properties.get("model_dir")

        # Unpack and preprocess input data for prediction -
        # text file with coma-separated indices of nodes that we want to get the prediction for
        # path_to_input_data = os.path.join(model_dir, "predict_input_data.pt")
        file = open(data['data'], "r")
        print("File with input data opened")
        lines = file.readlines()
        lines = [line.split(",") for line in lines]
        indices = [int(x) for y in lines for x in y]

        # Load data for the model
        input_model_data = torch.load(data)[0]
        print("Model loaded")
        x, edge_index = self.preprocess(input_model_data)
        print("Data loaded")

        # Predict
        model_output = self.inference(x, edge_index)
        pred = model_output.argmax(dim=1)
        # Return prediction for the desired nodes
        return pred[indices]

if __name__ == "__main__":
    handler = CustomGraphHandler()
    print(handler.handle({'data': 'predict_input_data.txt'}))
