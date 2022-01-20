import torch
import os
from ts.torch_handler.base_handler import BaseHandler

class CustomGraphHandler(BaseHandler):
    
    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        x = data.x.clone()
        edge_index = data.edge_index.clone()
        # y = data.y.clone()
        return torch.as_tensor(x, device=self.device), torch.as_tensor(edge_index, device=self.device) # , torch.as_tensor(y, device=self.device)


    def inference(self, x, edge_index):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(x, edge_index)
        return model_output


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """

        # Get path to data from extra files
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Unpack and preprocess input data for prediction - text file with coma-separated indices of nodes that we want to get the prediction for
        path_to_input_data = os.path.join(model_dir, "predict_input_data.pt")   
        file = open(path_to_input_data, 'r')
        lines = file.readlines()
        lines = [line.split() for line in lines]
        indices = [int(x) for y in lines for x in y]

        # Load data for the model
        input_model_data = torch.load(data)[0]
        x, edge_index = self.preprocess(input_model_data)
        
        # Predict
        model_output = self.inference(x, edge_index)
        pred = model_output.argmax(dim=1)
        # Return prediction for the desired nodes
        return pred[indices]
