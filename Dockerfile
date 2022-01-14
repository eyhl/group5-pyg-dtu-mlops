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
FROM python:3.8.12-bullseye

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         #python3.8 \
        ca-certificates && \
     rm -rf /var/lib/apt/lists/*

# RUN ln -s /usr/bin/python3.8 /usr/bin/python3

COPY ./requirements_docker.txt /.
COPY ./setup.py /.


# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install -r requirements_docker.txt --no-cache-dir && \
    #pip install setuptools && \
    rm get-pip.py

WORKDIR /root

# Installs pytorch and torchvision manually as they do not want to cooporate.
RUN pip download torch==1.10.1
RUN pip install torch*.whl 
RUN pip install torchvision==0.11.2

# Install PyG.
# RUN CPATH=/usr/local/cuda/include:$CPATH \
#  && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
#  && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install scipy

RUN pip install torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

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
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin -

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
RUN mkdir /root/project
WORKDIR /root/project
COPY src/ /root/project/src/
COPY models/ /root/project/models
COPY entrypoint.sh /root/project/entrypoint.sh
COPY .dvc/ /root/project/.dvc/
COPY data.dvc /root/project/data.dvc
COPY .git/ /root/project/.git/

ENV PYTHONPATH "${PYTHONPATH}:/root/project"

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["sh", "entrypoint.sh"]