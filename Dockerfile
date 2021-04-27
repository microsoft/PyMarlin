FROM mcr.microsoft.com/azureml/base:openmpi3.1.2-ubuntu18.04
RUN apt-get update


# create conda environment
RUN conda update -n base -c defaults conda -y
RUN conda create -n marlin python=3.8 -y
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc

#install torch latest
# Cuda toolkit other than 1.2 makes GPUs invisible. Base image issue
RUN conda install pytorch cudatoolkit=10.2 -c pytorch -y -n marlin

ADD . /workdir
WORKDIR /workdir

RUN /opt/miniconda/envs/marlin/bin/pip install -U -e . 
RUN /opt/miniconda/envs/marlin/bin/pip install -U -r requirements.txt

# Instructions to update docker image. (replace krishansubudhi with your dockerhub account name)
# https://krishansubudhi.github.io/development/2019/09/23/CreatingDockerImage.html

# In a VM, Build
# docker build --rm -t krishansubudhi/marlin:latest .

# Test
# docker run --gpus all -it -d -p 5000:5000 krishansubudhi/marlin:latest
# docker attach <guid from the above command>

# Push new image to dockerhub
# docker push krishansubudhi/marlin:latest