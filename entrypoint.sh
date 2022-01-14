#!/bin/sh
dvc pull
wandb login $1
python -u src/models/train_model.py $2