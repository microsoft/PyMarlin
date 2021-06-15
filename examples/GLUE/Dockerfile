FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04
RUN apt-get update


# create conda environment
RUN conda update -n base -c defaults conda -y
RUN conda create -n marlin python=3.8 -y
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc



ADD requirements.txt /workdir/
WORKDIR /workdir

RUN /opt/miniconda/envs/marlin/bin/pip install -U -r requirements.txt

# Instructions to update docker image. (replace krishansubudhi with your dockerhub account name)
# https://krishansubudhi.github.io/development/2019/09/23/CreatingDockerImage.html

# In a VM, Build
# docker build --rm -t krishansubudhi/marlin:latest .

# Test
# docker run --gpus all -it -d -p 5000:5000 krishansubudhi/marlin:latest
# docker ps 
# CONTAINER_ID=4dd751e87293 # replace 4dd751e87293 wit your container id
# docker cp ./src $CONTAINER_ID:/workdir
# docker attach $CONTAINER_ID

# Push new image to dockerhub
# docker login
# docker push krishansubudhi/marlin:latest