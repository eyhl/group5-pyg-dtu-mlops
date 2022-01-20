import argparse
from copy import deepcopy

import sklearn.manifold
import torch
import torchdrift
from torch_geometric.loader import DataLoader

from src.data.make_dataset import load_data
from src.models.model import load_checkpoint
from src.visualization.visualize import visualise_drift


def detect_drift(orig_dataloader, corrupt_im, feature_extractor, kernel):
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector(kernel=kernel)
    torchdrift.utils.fit(orig_dataloader, feature_extractor, drift_detector)
    features = feature_extractor(corrupt_im)
    features_np = features.detach().numpy()
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    mapper = sklearn.manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features_np)
    return score, p_val, base_embedded, features_embedded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ood', action='store_true')
    args = parser.parse_args()

    batch_size = 64
    # Load model
    model = load_checkpoint('./models/checkpoint.pt')
    feature_extractor = model
    orig_data = load_data('./data/', 'Cora')

    corrupt_data = deepcopy(orig_data)
    corrupt_data.x = torch.randint(0, 2, (2708, 1433), dtype=torch.float32)
    orig_dataloader = DataLoader([orig_data])

    kernels = ["Gaussian Kernel", "Exp Kernel", "Rational Quadratic Kernel"]
    results = {}
    for kernel_name in kernels:
        if kernel_name == "Gaussian Kernel":
            kernel = torchdrift.detectors.mmd.GaussianKernel()
        elif kernel_name == "Exp Kernel":
            kernel = torchdrift.detectors.mmd.ExpKernel()
        elif kernel_name == "Rational Quadratic Kernel":
            kernel = torchdrift.detectors.mmd.RationalQuadraticKernel()
        else:
            print("Unknown kernel")
            break
        results[kernel_name] = detect_drift(orig_dataloader,
                                            corrupt_data,
                                            feature_extractor,
                                            kernel,
                                            )
    visualise_drift(results, kernels, args.ood)
