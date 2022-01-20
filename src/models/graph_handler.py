import torch
import os
from ts.torch_handler.base_handler import BaseHandler


class CustomGraphHandler(BaseHandler):

    def __init__(self):
        self.initialized = False
        self.model = None
        self._context = None
        self.model_data = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)
        print("Model loaded")

        # model_path = "models/deployable_model.pt"
        # model_config_path = "src/config/exp1.yaml"
        # with open(model_config_path) as file:
        #     hparams = yaml.load(file, Loader=yaml.FullLoader)

        # model = GCNConv(hidden_channels=hparams["hidden_channels"],
        #                 num_features=hparams["num_features"],
        #                 num_classes=hparams["num_classes"],
        #                 dropout=hparams["dropout"])

        # model.load_state_dict(torch.load(model_path))
        # self.model = model.eval()
        # print("Loaded model:", model)

        # Load data supplied by --extra-files
        path_to_data = os.path.join(model_dir, 'data.pt')
        self.model_data = torch.load(path_to_data)[0]
        print("Data loaded.")

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
        file = open(data['data'], "r")
        print("File with input data opened")
        lines = file.readlines()
        lines = [line.split(",") for line in lines]
        indices = [int(x) for y in lines for x in y]

        return indices

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
        # Load data for the model
        indices = self.preprocess(data)

        # Predict
        model_output = self.inference(self.model_data.x, self.model_data.edge_index)
        pred = model_output.argmax(dim=1)
        # Return prediction for the desired nodes
        return pred[indices]

if __name__ == "__main__":
    handler = CustomGraphHandler()
    print(handler.handle({'data': 'predict_input_data.txt'}))
