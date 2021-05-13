# Installation
In this guide, we will share instructions on how to set up pymarlin in the following environments:
* Local/Dev Machine
* AzureML

## Local/Dev Machine
### Environment setup
    conda create -n pymarlin python=3.8
    conda activate pymarlin

### Install pytorch
[Latest documentation](https://pytorch.org/get-started/locally/)

    conda install pytorch cpuonly -c pytorch

### Install PyMarlin
You can install from our internal pip or alternatively install from source.

#### Install from pip

    pip install pymarlin

#### Install from source

    git clone https://github.com/microsoft/PyMarlin.git
    cd PyMarlin
    pip install -e .

## AzureML
Specify the pip package in a supplied conda_env.yml file.