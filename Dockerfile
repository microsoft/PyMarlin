FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04


#install torch latest

RUN conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c nvidia

ADD . /workdir
WORKDIR /workdir

RUN pip install pymarlin[plugins] --ignore-installed

Run pip install opacus

Run pip install Datasets