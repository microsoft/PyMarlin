# Optimizing with ORT + DeepSpeed
We have extended this example to be optimized with ORT+DeepSpeed! Starting from this scenario we will try to build common backend for both ORT and DeepSpeed.

## Noteworthy files
* [deepspeed_methods](deepspeed_methods): deepspeed utility methods and trainer / trainer backends.
* [model_ortds.py](model_ortds.py): module interface with config checks to enable ort+deepspeed
* [train_ortds.py](train_ortds.py): main train script that imports the above
* [aml/submit_ortds.py](aml/submit_ortds.py): azureml submit script (needs to be abstracted to upload data / not hard-code compute/ pull docker image built from pymarlin docker repo)

## Submitting
1. Create an AzureML workspace and place the config.json with subscription id, resouce group, workspace name into the aml folder.
2. Create a gpu cluster.
3. Adjust the values in submit_ortds.py to point to your new gpu cluster, and upload data. (make this autonomous?)
4. Upload preprocessed CNN/DailyMail from original README by uncommenting the lines to.
4. Install azureml-sdk with `pip install azureml-sdk`
5. From examples/summarization/aml, Submit job with `python submit_ortds.py` TODO: use common azureml submit.