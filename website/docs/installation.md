# Installation
In this guide, we will share instructions on how to set up pymarlin in the following environments:
* Local/Dev Machine
* Public AzureML
* Compliant AzureML (Rufous Stage 3)

## Local/Dev Machine
### Environment setup
    conda create -n pymarlin python=3.8
    conda activate pymarlin

### Install pytorch

[Latest documentation](https://pytorch.org/get-started/locally/)

    conda install pytorch cpuonly -c pytorch

## Install pymarlin
You can install from our internal pip or alternatively install from source.

### Install from pip package
This pip package is internal to microsoft employees currently. Hence the index-url needs to be specified. `artifacts-keyring` handles the authentication.

    pip install keyring artifacts-keyring #https://github.com/microsoft/artifacts-keyring

    pip install pymarlin --index-url https://o365exchange.pkgs.visualstudio.com/959adb23-f323-4d52-8203-ff34e5cbeefa/_packaging/ELR/pypi/simple/

### Install from source
Alternatively, to get the latest changes, build from the source itself

    git clone https://o365exchange.visualstudio.com/DefaultCollection/O365%20Core/_git/ELR
    git checkout -b u/elr/refactor
    cd ELR\sources\dev\SubstrateInferences\pymarlin
    pip install -r requirements.txt

#### Option 1 : pip install 
https://medium.com/@arocketman/creating-a-pip-package-on-a-private-repository-using-setuptools-fff608471e39


    pip install .
    cd .. 
    python -c 'import pymarlin; print(pymarlin.__path__)'
    python

    Hello World

#### Option 2 : PYTHONPATH
Do this when making changes to the library. Suitable for core folks who work on improving the library.

For windows cmd,

    set PYTHONPATH=<full path to ELR\sources\dev\SubstrateInferences\pymarlin>

## Docker Image for Public AzureML
We have an MSFT-tenant ACR that hosts an AzureML compatible docker image with pymarlin. Here are the instructions for how to link to it in your public AzureML experiment. 

```
from azureml.core.environment import Environment
# Create the environment
pymarlinenv = Environment(name="pymarlinenv")
# Enable Docker and reference an image
pymarlinenv.docker.enabled = True
# Set the container registry information, password subject to change, reach out to josleep@ for latest.
pymarlinenv.docker.base_image = "pymarlin:latest"
pymarlinenv.docker.base_image_registry.address = "elrsubstrate.azurecr.io"
pymarlinenv.docker.base_image_registry.username = "elrsubstrate"
pymarlinenv.docker.base_image_registry.password = "reach out to josleep@ for password"
```

## Pip package for Compliant AzureML (Rufous Stage 3)
When experimenting in Compliant AzureML, you can add any packages in the Polymer package feed to your conda environment. See an example in our own repo [here](https://o365exchange.visualstudio.com/O365%20Core/_git/ELR?path=%2Fsources%2Fdev%2FSubstrateInferences%2Fpymarlin_Scenarios%2Ftest_hfseqclass%2Fconda_env.yaml&version=GBu%2Fjosleep%2Frufous-pipeline&line=9&lineEnd=10&lineStartColumn=1&lineEndColumn=105&lineStyle=plain&_a=contents) for how to add pymarlin as a dependecy in your module's conda environment file.

## Docker Image for Compliant AzureML (Rufous Stage 3)
polymerprod.azurecr.io/polymercd/prod_official/pymarlin:1.0.0