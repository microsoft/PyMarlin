# Optimizing with ORT + DeepSpeed
We have extended this example to be optimized with ORT+DeepSpeed! Starting from this scenario we will try to build common backend for both ORT and DeepSpeed.

## Noteworthy files
* [deepspeed_methods](deepspeed_methods): deepspeed utility methods and trainer / trainer backends.
* [model_ortds.py](model_ortds.py): module interface with config checks to enable ort+deepspeed
* [train_ortds.py](train_ortds.py): main train script that imports the above
* [aml/submit_ortds.py](aml/submit_ortds.py): azureml submit script (needs to be abstracted to upload data / not hard-code compute/ pull docker image built from pymarlin docker repo)

## Submitting
1. Install azureml-sdk and create an AzureML workspace, great [instructions on both here](https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/installation).
2. Write out the config.json for the workspace with [write_config()](https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/workspace#helpful-methods)
3. Create a gpu cluster in the workspace, for more info go [here](https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/compute-targets#creating-compute-targets)
4. Adjust the values in submit_ortds.py to point to your new gpu cluster.
5. Upload preprocessed CNN/DailyMail from original README by uncommenting line 48 and point to local path.
6. From examples/summarization/aml, Submit job with `python submit_ortds.py`