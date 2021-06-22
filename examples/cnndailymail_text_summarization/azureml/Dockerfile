# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

USER root:root

# ------------------------------------------------------------------------------------------------ #
# Environment variables
# ------------------------------------------------------------------------------------------------ #

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH "/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

ENV STAGE_DIR=/root/gpu/install \
    CUDA_HOME=/usr/local/cuda \
    CUDNN_HOME=/usr/lib/x86_64-linux-gnu \
    CUDACXX=/usr/local/cuda/bin/nvcc

RUN mkdir -p $STAGE_DIR

# ------------------------------------------------------------------------------------------------ #
#
# ------------------------------------------------------------------------------------------------ #

RUN apt-get -y update && \
    apt-get --assume-yes --no-install-recommends install \
    build-essential \
    autotools-dev \
    curl \
    wget \
    openssh-server \
    openssh-client \
    tmux \
    vim \
    sudo \
    g++ \
    gcc \
    git \
    bc \
    tar \
    bash \
    pbzip2 \
    pv bzip2 \
    cabextract \
    dos2unix \
    less \
    unzip \
    pdsh \
    pssh \
    nfs-common \
    libfuse-dev \
    htop iftop iotop rsync iputils-ping \
    net-tools && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------------------------ #
# Conda, python and pip
# ------------------------------------------------------------------------------------------------ #

ARG PYTHON_INSTALL_VERSION=3.7
ARG PIP_INSTALL_VERSION=20.1.1

ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh    && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

RUN conda install -y python=$PYTHON_INSTALL_VERSION
RUN conda install -y pip=$PIP_INSTALL_VERSION

# ------------------------------------------------------------------------------------------------ #
# IB user space libs
# ------------------------------------------------------------------------------------------------ #

RUN apt-get update && apt-get install -y --no-install-recommends pciutils ibutils ibverbs-utils rdmacm-utils infiniband-diags perftest librdmacm-dev && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------------------------ #
# OPENMPI
# ------------------------------------------------------------------------------------------------ #

ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION_STRING=${OPENMPI_BASEVERSION}.0
RUN cd ${STAGE_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION_STRING}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION_STRING} && \
    ./configure  --enable-orterun-prefix-by-default && \
    make uninstall && \
    make -j"$(nproc)" install && \
    # Sanity check:
    test -f /usr/local/bin/mpic++ && \
    ldconfig && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION_STRING}
ENV PATH=/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# ------------------------------------------------------------------------------------------------ #
# cmake
# ------------------------------------------------------------------------------------------------ #

ENV CMAKE_VERSION=3.16.4
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-Linux-x86_64/bin:${PATH}

WORKDIR /workspace

# ------------------------------------------------------------------------------------------------ #
# PyTorch
# ------------------------------------------------------------------------------------------------ #

RUN pip install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# ------------------------------------------------------------------------------------------------ #
# Tensorflow, tensorboard, tensorboardX
# ------------------------------------------------------------------------------------------------ #

RUN pip install --no-cache-dir tensorflow tensorboard tensorboardX

# ------------------------------------------------------------------------------------------------ #
# pymarlin install
# ------------------------------------------------------------------------------------------------ #

RUN pip install --no-cache-dir pymarlin[plugins]

# ------------------------------------------------------------------------------------------------ #
# Install Apex
# ------------------------------------------------------------------------------------------------ #

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
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

# ------------------------------------------------------------------------------------------------ #
# Install ORT
# ------------------------------------------------------------------------------------------------ #

RUN git clone https://github.com/microsoft/onnxruntime.git &&\
    cd onnxruntime &&\
    git submodule update --init --recursive && \
    python tools/ci_build/build.py \
        --cmake_extra_defines \
            ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER` \
        --config Release \
        --enable_training \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --update \
        --parallel \
        --build_dir build \
        --build \
        --build_wheel \
        --skip_tests &&\
    pip install build/Release/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime /opt/cmake
