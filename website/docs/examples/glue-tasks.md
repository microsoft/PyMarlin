# Glue Tasks

You can use the `pymarlin` library to easily benchmark your models for the GLUE tasks.
The following walkthrough references the source code located [here](https://github.com/microsoft/PyMarlin/tree/main/examples/glue).

This walkthrough will be focused on the GLUE RTE task, run on CPU, although the source code is setup to run 8 tasks (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE) and can also be run on a VM or Azure ML with distributed training.

    Download the RTE dataset and setup as below:
    root
    |-- RTE
        |-- train.tsv
        |-- dev.tsv

## Data Preprocessing
`config.yaml` contains all the arguments needed for data preprocessing. `data.py` is the script for data preprocessing.
```python
# data-module arguments
dmod:
    input_dir : "" # provide path to data directory here, it should only contain files
    output_dir : ""

# data-processor args
dproc:
    task: null # specify the task name (e.g. RTE)
    max_seq_len: 128
    no_labels: False
    set_type: null # specify "train" or "dev" depending on which dataset is being preprocessed
    tokenizer: "bert-base-uncased" # huggingface tokenizer name
```

You can override the default values in the config through CLI:

    Command for train.tsv:
        $ python data.py --dmod.input_dir "RTE" --dmod.output_dir "processed_data/train" --dproc.task "RTE" --dproc.set_type "train"
    Command for dev.tsv:
        $ python data.py --dmod.input_dir "RTE" --dmod.output_dir "processed_data/dev" --dproc.task "RTE" --dproc.set_type "dev"

    root
    |-- processed_data
        |-- train
            |-- train.tsv (pickle dump)
        |-- dev
            |-- dev.tsv

Data preprocessing consists of the following steps:
1. Reading the input tsv file to create `InputExample`. This is done by `glue_processors.RTEProcessor`.
    ```python
        class RteProcessor(GLUEBaseProcessor):
            """Processor for the RTE data set (GLUE version)."""
            def __init__(self, args):
                super().__init__(args)
                
            def get_labels(self):
                """See base class."""
                return ["entailment", "not_entailment"]

            def _create_examples(self, lines, set_type):
                """Creates examples for the training and dev sets."""
                examples = []
                for (i, line) in enumerate(lines):
                    if i == 0:
                        continue
                    guid = i
                    text_a = line[1]
                    text_b = line[2]
                    label = line[-1] if set_type != "test" else "entailment"
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                return examples
    ```
2. Tokenizing the text data to create features required for the Bert model (convert `InputExample` to `InputFeatures`). This is done by the `Featurizer` processor written in data.py.
Both `RTEProcessor` and `Featurizer` are based on `pymarlin.core.data_interface.DataProcessor` which provides an additional functionality for python multiprocessing. The `process` method in these data processors executes the logic written by the user. The optional `analyze` method is intended to compute user specified data stats.
    ```python
        class Featurizer(DataProcessor):
            def __init__(self, args, labels_list, tokenizer=None):
                self.args = args
                self.label_map = {label: i for i, label in enumerate(labels_list)} if None not in labels_list else None
                self.tokenizer = tokenizer

            def process(self, examples, output_path, save_features=False):
                self.features = []
                for example in examples:
                    tokens =  tokenizer(example.text_a,
                                        example.text_b,
                                        max_length=self.args.max_seq_len,
                                        padding='max_length',
                                        truncation=True,
                                        return_token_type_ids=True,
                                        return_tensors='pt')

                    if example.label is not None:
                        if self.label_map is not None: # classification task
                            label = self.label_map[example.label]
                        else: # regression task data processor returns labels list [None]
                            label = float(example.label)
                    else: # labels not provided (only inference)
                        label = None
                    self.features.append(InputFeatures(tokens.input_ids.squeeze().tolist(),
                                                tokens.attention_mask.squeeze().tolist(),
                                                tokens.token_type_ids.squeeze().tolist(),
                                                label))
                if save_features:
                    with open(output_path, 'wb') as f:
                        pickle.dump(self.features, f)
                return self.features

            def analyze(self):
                logger.info(f"# of features processed = {len(self.features)}")
    ```
3. Setup your PyTorch Dataset that converts above features to tensors ready to be consumed by the model during training. This is done by `TaskDataset`.
    ```python
        class TaskDataset(Dataset):
            def __init__(self, datapath):
                self.datapath = datapath
                self.load_features()

            def load_features(self):
                with open(self.datapath, 'rb') as f:
                    self.features = pickle.load(f)
                    
            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                feature = self.features[idx]
                return self._create_tensors(feature)
            
            def _create_tensors(self, feature):
                input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
                input_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
                segment_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
                tensor_dict = {'input_ids':input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids}
                if feature.label is not None:
                    if type(feature.label) == int:
                        label_id = torch.tensor(feature.label, dtype=torch.long)
                    else:
                        label_id = torch.tensor(feature.label, dtype=torch.float)
                    tensor_dict.update({'labels': label_id})
                return tensor_dict
    ```
4. `pymarlin.core.data_interface.DataInterface` is a template to orchestrate the execution of various processors that user implements and also retrieves the implemented PyTorch Datasets (used to feed the DataLoader during training).
    ```python
        class TaskData(DataInterface):
            def __init__(self, args):
                super().__init__()
                self.args = args

            def create_dataset(self, datapath):
                return TaskDataset(datapath)
                
            def get_train_dataset(self, trainpath):
                return self.create_dataset(trainpath)
                
            def get_val_dataset(self, valpath):
                return self.create_dataset(valpath)
            
            def get_test_dataset(self, testpath):
                return self.create_dataset(testpath)
    ```

## Training
`config.yaml` also contains all the arguments needed for training. `train.py` is the script for training. The model we will finetune is HuggingFace's `bert-base-uncased`, specified through `model.hf_model`. 

    Command for training & validation:
        $ python train.py --tmod.task "RTE" --tmod.trainpath "processed_data/train" --tmod.valpath "processed_data/dev" --tmod.output_dir "training_output"
    Command for inference only:
        $ python train.py --tmod.task "RTE" --tmod.valpath "processed_data/dev" --tmod.output_dir "inference_output"

    The difference is that when doing inference/validation only, it does not need `tmod.trainpath`.

`pymarlin.core.module_interface.ModuleInterface` is the building block that interacts with various components of `pymarlin.core` library. This is the block that the user should implement for their task. For GLUE, the implemented TrainModule is called `Recipe`. Note that the module inherits from `torch.nn` so it can also act as the model you would like to train.

The key parts of `Recipe`:

1. Setting up dataloaders for training and validation. For GLUE, the training dataloader is actually a generator, wrapped by PyTorch Dataloader. We implemented a generator that yields batches per input file, which may be useful for non-GLUE tasks that may lead to Out of Memory errors if the entire dataset is loaded. This is not required, so you are free to use any PyTorch Dataloader that works for your data. 
    ```python
        def get_train_dataloader(self, sampler, batch_size):
            dl = FilesGenDataloader(self.args, self.datamodule, "train")
            total_datacount = dl.get_datacount()
            self.logger.info(f"Total training samples = {total_datacount}")
            dl_gen = dl.get_dataloader(total_datacount, sampler, batch_size)
            return dl_gen
    ```
    The validation dataloader returns a dictionary of dataloaders where the key is the name of the file. This is useful for tasks which may need to do post-processing after validation (and all_gathering in the case of distributed inferencing) with the original data file. 
    ```python
        def get_val_dataloaders(self, sampler, batch_size):
            dl = FilesDictDataloader(self.args, self.datamodule, "val")
            total_datacount = dl.get_datacount()
            self.logger.info(f"Total validation samples = {total_datacount}")
            dl_dict = dl.get_dataloader(sampler, batch_size)
            return dl_dict
    ```

2. Provide optimizers and schedulers.
    ```python
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight'] # bert also has LayerNorm.bias
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        ### Using Huggingface optimizer & scheduler for glue ###
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.max_lr, eps=1e-8, betas=(0.9, 0.999))
        training_steps = estimated_global_steps_per_epoch * epochs
        warmup_steps =  self.args.warmup_prop * training_steps
        self.logger.debug(f"# of warmup steps = {warmup_steps}, # of training steps = {training_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )
    ```

3. Setting up model and related items such as config as needed.
    ```python
        self.model_config = AutoConfig.from_pretrained(self.args.model_args.hf_model)
        if self.args.task.lower() == 'sts-b': # STS-B is a regression task
            self.model_config.num_labels = 1
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_args.hf_model, config=self.model_config)
    ```

4. Specify `train_step` and `val_step` based on the model you setup.
    ```python
        def train_step(self, global_step, batch, device):
            inputs = self._inputs_to_device(batch, device)
            outputs = self.model.forward(**inputs)
            loss = outputs.loss
            return loss
    ```
    For GLUE, we set up our `TaskDataset` such that it creates a dictionary of tensors. This allows use to directly pass in the inputs to the model. The outputs of a HuggingFace model is a Dataclass. We recommend the practice of creating model outputs as a Dataclass so it's easily understood and accessed. While `train_step` only returns the loss, you can choose which outputs to return in `val_step` which can be used by callbacks later on.
    ```python
        def val_step(self, global_step, batch, device):
            inputs = self._inputs_to_device(batch, device)
            outputs = self.model.forward(**inputs)
            if outputs.loss is not None:
                return outputs.loss, outputs.logits, inputs['labels']
            else:
                return outputs.logits
    ```

5. Use the callbacks to do post-processing. The most important callback for GLUE tasks is  `on_end_val_epoch` which is where we compute metrics for the task. You can use `global_stats` to easily create AML or Tensorboart charts.
    ```python
        def on_end_val_epoch(self, global_step, *inputs, key="default"):
            """
            args contains all values returned by val_step all_gathered over all processes and all steps
            """
            if not self.args.no_labels:
                losses, logits, labels = inputs
                losses = torch.stack(losses).mean().item()
                global_stats.update(key+'/val_loss', losses, frequent=False)
                labels = torch.cat(labels).cpu().numpy()
                logits = torch.stack(logits)
                preds = torch.argmax(logits, dim=-1).view(-1).cpu().numpy()
                assert len(preds) == len(labels)
                metrics = glue_dict[self.args.task.lower()]["metric"](labels, preds)
                for k in metrics:
                    global_stats.update(key+'/val_'+k, metrics[k])
    ```













