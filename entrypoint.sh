#!/bin/sh
#dvc pull
wandb login $YOUR_API_KEY
python -u src/models/train_model.py