# Germ Eval NER task

Ashwin Srinivasan

This example will walk you through executing the plugin for Germ eval NER task.

For more detailed understanding of how you can use NER plugin please refer to [this](https://microsoft.github.io/PyMarlin/docs/plugins/hf_ner)

## Dataset format

NER plugin expects the input to be a TSV or CSV with 2 columns. A column with the text sentences followed by a column with the labels for the tokens in the sentence.'Sentence':'who is harry', 'Slot': 'O O B-contact_name'

For GermEval dataset we have already modified and provided the dataset along with this example. You will find the train file under train_germ and dev file under val_germ.

## Running on VM
```
conda create -n pymarlin
conda activate pymarlin
pip install -r requirements.txt
```

Moving data : 

scp -r -P $port .\examples\NER_Plugin\  $user@${machine}:/home/$user
ssh $user@$machine -p $port

Running on CLI would be as simple as:

```
python test.py --data.train_filepath ./train_germ/train.tsv --data.val_filepath ./val_germ/dev.tsv --config_path config_germ.yaml
```

Running with multi-GPU:

```
python -m torch.distributed.launch --nproc_per_node 4 test.py --data.train_filepath ./train_germ/train.tsv --data.val_filepath ./val_germ/dev.tsv --config_path config_germ.yaml --trainer.backend ddp-amp
```
Results:

    tensorboard --logdir logs

Tunnel to view tensorboard UI: (if using VM):

    ssh -N -f -L 127.0.0.1:6006:127.0.0.1:6006  $user@${machine} -p $port

View Tensorboard UI:

    http://localhost:6006/

## Running on Azure ML

A notebook has been provided along with this titled 'GermEvalAML.ipynb' , once you have a valid azure workspace , resource group and compute replace the placeholders in the notebook and you should be able to submit a script to AML.

## Mode checkpoint extraction + Inference

You may want to further use this model checkpoint for inference or use it in your project. The instructions are [here](https://microsoft.github.io/PyMarlin/docs/plugins/hf_ner) under evaluation section. Further the notebook includes a inference section with the relevant code.
