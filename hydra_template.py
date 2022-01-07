# imports
import hydra
'''
Usage:
$ python hydra_template.py experiment=exp1
$ python hydra_template.py experiment=exp2
...
'''
@hydra.main(config_path="config", config_name="default_config")
def train(config):
    cfg = config.experiment
    lr = cfg.hyperparams.lr
    batch_size = cfg.hyperparams.batch_size

if __name__ == "__main__":
    train()
