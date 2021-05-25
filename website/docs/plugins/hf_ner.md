# Named Entity Recognition with HuggingFace models

We designed this plugin to allow for out-of-the-box training and evaluation of HuggingFace models for NER tasks. We provide a golden config file (config.yaml) which you can adapt to your task. This config will make experimentations easier to schedule and track.

## Step by step with GermEval dataset

We will go through how to adapt any dataset/task for pymarlin and how to setup the plugin. For this purpose we will use the GermEval dataset - this is a dataset with German Named Entity annotation , with data sampled from German Wikipedia and News Corpora. For more granular information and raw dataset please refer [here](https://sites.google.com/site/germeval2014ner/data)

Following HuggingFace documentation for preliminary data clean up we use their preprocess script to clean up the original dataset. These can be run in jupyter Notebook.

```python
!wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
!grep -v "^#" NER-de-train.tsv| cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
!grep -v "^#" NER-de-dev.tsv| cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
!python preprocess.py train.txt.tmp 'bert-base-multilingual-cased' '128' > train.txt
!python preprocess.py dev.txt.tmp 'bert-base-multilingual-cased' '128' > dev.txt
!cat train.txt dev.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

## Dataset format

NER plugin expects the input to be a TSV or CSV with 2 columns. A column with the text sentences followed by a column with the labels for the tokens in the sentence. For example: 'Sentence': 'who is harry', 'Slot': 'O O B-contact_name'

For GermEval dataset below we show how to modify to format expected by plugin.

```python
import csv
def txt2tsv(filename, outfile):
  outfile = open(outfile, "w")
  f = open(filename, "r")
  lines = f.readlines()
  sentence = []
  labels = []
  tsv_writer = csv.writer(outfile, delimiter='\t')
  tsv_writer.writerow(['Sentence', 'Slot'])
  for line in lines:
    line = line.strip()
    if line:
      row = line.split(' ')
      sentence.append(row[0])
      labels.append(row[1])
    else:
      sent = ' '.join(sentence)
      lab = ' '.join(labels)
      tsv_writer.writerow([sent, lab])
      sentence = []
      labels = []

txt2tsv("dev.txt", "dev.tsv")
txt2tsv("train.txt", "train.tsv")
```

The dataset would now look like this:

![Dataset](images/hfner/ner_dataset_mod.png)


## Golden yaml config

pymarlin leverages yaml files for maintaining experiment parameters. For this German Evaluation dataset we provide a golden config `config_germ.yaml`. 

```python
# data_processor args
data:
    train_dir : null
    val_dir : null
    labels_list: [B-LOC, B-LOCderiv, B-LOCpart, B-ORG, B-ORGderiv, B-ORGpart, B-OTH, B-OTHderiv,
        B-OTHpart, B-PER, B-PERderiv, B-PERpart, I-LOC, I-LOCderiv, I-LOCpart, I-ORG, I-ORGderiv,
        I-ORGpart, I-OTH, I-OTHderiv, I-OTHpart, I-PER, I-PERderiv, I-PERpart, O]
    max_seq_len: 128
    pad_label_id: -100
    has_labels: True
    tokenizer: "bert-base-multilingual-cased"
    file_format: "tsv"
    label_all_tokens: False

# model arguments
model:
    model_name: "bert"
    encoder_key: "bert"
    hf_model: "bert-base-multilingual-cased"
    model_file: "pytorch_model.bin"
    model_config_file: "config.json"
    model_path: null
    model_config_path: null

# module_interface arguments
module:
    operation: "train"
    tr_backend: "singleprocess"
    output_dir: null
    max_lr : 0.00003 # Maximum learning rate.
    warmup_prop: 0.1
    has_labels: True

# trainer arguments
trainer:
    train_batch_size: 32 # Training global batch size.
    val_batch_size: 16 # Validation global batch size.
    epochs: 25 # Total epochs to run.
    gpu_batch_size_limit : 8 # Max limit for GPU batch size during training.
    clip_grads : True # Enable or disable clipping of gradients.
    use_gpu: True # Enable or disable use of GPU.
    max_grad_norm: 1.0 # Maximum value for gradient norm.
    writers: ['stdout', 'aml', 'tensorboard'] # List of all the writers to use.
    disable_tqdm: True
    log_level: "DEBUG"
```

## Training

Next we need a orchestrating script to initialize the plugin and start training. Assume the script test.py. It will contain the following.

```python
from pymarlin.plugins import HfNERPlugin
plugin = HfNERPlugin()

plugin.setup()
plugin.trainer.train()
plugin.trainer.validate()
```

We can now schedule a run locally using CLI , modify to point to the train and validation directory appropriately :

```python
python test.py --data.train_dir ./train_germ --data.val_dir ./val_germ --config_path config_germ.yaml
```