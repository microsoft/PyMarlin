# Covid-19 Text Sentiment Classification

## Instructions
1. Install requirements
        pip install -r requirements.txt
2. Download data from kaggle
        Ref: https://github.com/Kaggle/kaggle-api
        Get your credentials file from kaggle here: C:\Users\<user>\.kaggle\kaggle.json
        kaggle datasets download -d datatattle/covid-19-nlp-text-classification
3. Install pymarlin library
        pip install pymarlin
        or
        $env:PYTHONPATH=<pymarlin repo path>
4. Set working directory
        cd pymarlin_scripts
5. Prepare data
        python data.py
6. Train
        python train.py [--trainer.max_train_steps_per_epoch 2]

## Running AzureML
You can use [`examples/classification/azureml/azureml_submit.py`](https://github.com/microsoft/PyMarlin/blob/main/examples/classification/azureml/azureml_submit.py)
to submit examples to run on Azure ML.

For example:

```bash
cd examples/classification/azureml/
python azureml_submit.py --backend ddp-amp --process_count 2 
--subcription_id <your azure sub id> --resource_group <rg> --workspace_name <azureml workspace>
```

See `examples/classification/azureml/azureml_submit.py -h` for more options.

**Note.** Submitting to AzureML requires setting up an AzureML workspace. See [Azure ML CheatSheet](https://aka.ms/aml/cheatsheet) for more details.