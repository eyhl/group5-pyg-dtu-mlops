import torch
import pytest
import hydra
import os.path
from src.data.make_dataset import load_data

@hydra.main(config_path="../config", config_name="default_config.yaml")
@pytest.mark.skipif(not os.path.exists("data/raw/ind.cora.x"), reason="Data files not found")
def test_data_length():
    # Load data
    orig_cwd = hydra.utils.get_original_cwd()
    data = load_data(orig_cwd + "/data/", name="Cora")

    
