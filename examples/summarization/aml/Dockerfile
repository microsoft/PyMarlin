# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04
# NCCL 2.7.8

USER root:root

# ------------------------------------------------------------------------------------------------ #
# Environment variables
# ------------------------------------------------------------------------------------------------ #

ENV STAGE_DIR=/root/gpu/install \
    CUDA_HOME=/usr/local/cuda \
    CUDNN_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc

RUN mkdir -p $STAGE_DIR

# ------------------------------------------------------------------------------------------------ #
# Install python and pip
# ------------------------------------------------------------------------------------------------ #

ARG PYTHON_INSTALL_VERSION=3.7.7
ARG PIP_INSTALL_VERSION=20.1.1

RUN conda install -y python=$PYTHON_INSTALL_VERSION
RUN conda install -y pip=$PIP_INSTALL_VERSION

# ------------------------------------------------------------------------------------------------ #
# CMAKE
# ------------------------------------------------------------------------------------------------ #

#ENV CMAKE_VERSION=3.20.3
#RUN cd /usr/local && \
#    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
#ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-Linux-x86_64/bin:${PATH}
#
#WORKDIR /workspace

RUN conda install cmake

# ------------------------------------------------------------------------------------------------ #
# PyTorch
# ------------------------------------------------------------------------------------------------ #

RUN pip install --no-cache-dir torch torchvision torchaudio

# ------------------------------------------------------------------------------------------------ #
# PyMarlin Packages
# ------------------------------------------------------------------------------------------------ #

RUN pip install --no-cache-dir pymarlin[plugins]


# ------------------------------------------------------------------------------------------------ #
# Install ORT
# ------------------------------------------------------------------------------------------------ #

RUN pip install --pre onnxruntime-training -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_cu102.html
RUN pip install torch-ort

# ------------------------------------------------------------------------------------------------ #
# Install apex
# ------------------------------------------------------------------------------------------------ #

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    export TORCH_CUDA_ARCH_LIST="Volta;Turing;Kepler+Tesla" &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# ------------------------------------------------------------------------------------------------ #
# Install DeepSpeed
# ------------------------------------------------------------------------------------------------ #

RUN git clone https://github.com/microsoft/DeepSpeed.git /DeepSpeed
RUN cd /DeepSpeed && \
    git config pull.ff only && \
    git pull && \
    git checkout master && \
    pip install -v . && \
    ds_report