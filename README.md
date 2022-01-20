PyTorch Geometric for Classification of Scietific Papers
==============================
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/tests.yml/badge.svg)
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/coverage.yml/badge.svg)
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/flake8.yml/badge.svg)
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/isort.yml/badge.svg)
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/mypy.yml/badge.svg)
<br/>
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

This repository contains the project work carried out by group 5 in the MLOps course taught at DTU ([course website](https://skaftenicki.github.io/dtu_mlops/)). Group 5 consists of: Eigil Y. H. Lippert, Kasia Otko, Lenka Hýlová and Sara D. Nielsen (see contributors list for individual github pages). 

1. **Overall goal:**
The   goal   is   to   classify   scientific   papers   into   seven   research areas based on a citation network and a dictionary describing content of the paper.
2. **Framework:**
For this project the [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#) ecosystem was used.   It   implements   neural   network
layers specific for graphs and defines convolutional operations on the graphs.
3. **Data:**
The Cora dataset consisting of 2,708 scientific publications   classified   among   seven   classes.   The   citation network consists of 10,556 links. Each publication is described by   a   binary   label   indicating   the   absence/presence   of   the corresponding word from the dictionary, which consists of 1,433 unique   words.   This   data   is   [publicly   available](https://deepai.org/dataset/cora). We will load it using PyTorch-Geometric interface.
4. **Deep learning models used?**
We are using Graph   Neural   Network   models   and   incorporating different Graph Convolutional Network (GCN) layer, implemented in Pytorch Geometric. As the focus of the project is MLOps, the model is not extremely complex, but it is able to classify the papers to a satisfyingly high degree. 

## Project flowchart
![Alt text](reports/figures/flowchart.png?raw=true "Flowchart")


## WandB report:
See the following overview report of the model performance: [Overview](https://wandb.ai/group5-dtumlops/group5-pyg-dtumlops/reports/Overview-of-project-results--VmlldzoxNDYyODk2?accessToken=6sjiecvilemd7q8en7ln598w1kom8bmnup0fsk7xka9e18add4pkvf9l4r4miq5c)<br/>
And the hyperparameter sweep experiments: [Experiments](https://wandb.ai/group5-dtumlops/group5-pyg-dtumlops/reports/Hyperparameter-sweep--VmlldzoxNDYzMDY1?accessToken=50527c4puh8c7addch7kqu5tsm4mswulgh8kad8kz4b13ytlyng66zapnjauhq04)


## Reproduce using the newest build (Docker image):
The newest build of the repo is provided as an docker image stored on google cloud. Image can be pulled from the Google Cloud Container with the following command:
```bash
docker pull gcr.io/linear-rig-337909/group5_proj_cpu_container:latest
```

## How to install
Installing the project on your machine should be straighforward although Pytorch Geometric can cause some trouble. Clone the repo:
```bash
git clone https://github.com/eyhl/group5-pyg-dtu-mlops.git
```
Install requirements, preferably in seperate virtual environment:
```bash
pip install -r requirements.txt
```

## How to run
Running the training locally can be done with calling the `train_model.py` from the terminal:
```bash
python src/models/train_model.py
```
And to predict with the model use
```bash
python src/models/predict_model.py
```

# For developers:
In order to submit jobs to run on the cloud using WandB you have to do the following. First define the relevant environment variables, such as an WandB API key (either in the terminal, or set as a static environment variable in your .bashrc file):
```bash
export WANDB_API_KEY=***********************
export JOB_NAME=JOB_NAME=grp5_cpu_job_$(date +%Y%m%d_%H%M%S)
export IMAGE_URI=gcr.io/linear-rig-337909/group5_proj_cpu_container:latest
export REGION=europe-west1
```
And then run:
```bash
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
$WANDB_API_KEY \
experiment.hyperparams.lr=0.02
```

for changing a single hyperparameter or 

```bash
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
$WANDB_API_KEY \
experiment=src/config/experiment/exp1.yml
```

If you are running different experiments. Note that the `experiment=...` argument has to be provided. If you are just debugging you can pass `experiment.hyperparams.load_model_from=/models/`


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the project and running it locally
    ├── requirements-docker.txt   <- The requirements file for running the docker file as some packages are installed individually
    │                         in the Dockerfile 
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │  
    │   ├── config         <- Experiment configuration files to be used with hydra
    │   │   ├── experiment <- Various expriment setups
    │   │   │   └── exp1.yaml
    │   │   └── default_config.yaml
    │   │  
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to define arcitectur, train models, use trained models to make
    │   │   │                 predictions and for cprofiling the model scripts
    │   │   ├── predict_model.py
    │   │   ├── model.py
    │   │   ├── train_model_cprofile_basic.py
    │   │   ├── train_model_cprofile_sampling.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
