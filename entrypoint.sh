#!/bin/sh
#dvc pull
key=$(<decrypted-data.txt)
wandb login $key
python -u src/models/train_model.py