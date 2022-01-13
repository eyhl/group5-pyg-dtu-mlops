# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile-gpu
FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu18.04

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python3-distutils && \
     rm -rf /var/lib/apt/lists/*

#RUN ln -s /usr/bin/python3.9-dev /usr/bin/python3

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

WORKDIR /root

# Installs pytorch and torchvision.
RUN pip install torch==1.0.0 torchvision==0.2.1

# Install PyG.
RUN CPATH=/usr/local/cuda/include:$CPATH \
 && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
 && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install scipy

RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install torch-geometric

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
RUN mkdir /root/src
COPY src/models/train_model.py /root/src/train_model.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "train_model.py"]