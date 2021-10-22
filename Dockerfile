FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04

# install torch latest
RUN conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c nvidia

# install latest pymarlin published to pip with extra plugin dependencies like transformers (check setup.py)
RUN pip install pymarlin[plugins] --ignore-installed

# install opacus
RUN pip install opacus

# install deepspeed
RUN pip install deepspeed

# install ORT
RUN pip install onnxruntime-training -f https://download.onnxruntime.ai/onnxruntime_stable_cu111.html
RUN pip install torch-ort --ignore-installed
RUN python -m torch_ort.configure