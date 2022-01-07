group5-pyg-dtu-mlops
==============================

Group 5 in mlops at DTU working on pytorch geometric

1. **Overall goal of the project?**
The   goal   is   to   classify   scientific   papers   into   seven   research areas based on a citation network and a dictionary describing content of the paper.
2. **What framework are you going to use?**
Since we are going to work with graph structures, we will use PyTorch-Geometric   ecosystem.   It   implements   neural   network
layers specific for this type of data.
3. **How do you intend to include the framework into your project?**
We will use this ecosystem to load and transform the data and build Graph Neural Network model architecture.
4. **What data are you going to run on?**
We will start with a Cora dataset consisting of 2,708 scientific publications   classified   among   seven   classes.   The   citation network consists of 10,556 links. Each publication is described by   a   binary   label   indicating   the   absence/presence   of   the corresponding word from the dictionary, which consists of 1,433 unique   words.   This   data   is   [publicly   available](https://deepai.org/dataset/cora). We will load it using PyTorch-Geometric interface.
5. **What deep learning models do you expect to use?**
We are going to create our own deep learning model. We will work   with   Graph   Neural   Network   models   and   incorporate different   GNN   layers   implemented   in   PyTorch-Geometric, starting with Graph Convolutional Network (GCN) layer.

Project Organization
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
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
