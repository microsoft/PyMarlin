# NER Token Classification

This is an example explaining entire pipeline for NER task using pymarlin library. You can bring your own data and write a similar NER task for your dataset.

## Configs - YAML and Parsing

For ease of use we have configs passed in as YAML files. 
In this case we use the config file : config_prod.yaml included with example code.

Snippet of config:

```python
filename: "ner_dataset.csv"
dist: False
trainRatio: 0.8
tokenizer_type: "bert"
max_seq_length: 128
pad_label_id: -100

tmgr:
    train_batch_size: 32
    val_batch_size: 32 # Validation global batch size.
    epochs: 2 # Total epochs to run.
    gpu_batch_size_limit : 32 # Max limit for GPU batch size during training.
    disable_tqdm : False
    writers: ["aml", "tensorboard"]
```

This config can be read in like below : 

```python
#Create arg parser and read config
parser = CustomArgParser(log_level='DEBUG', default_yamlfile="config_prod.yaml")
config = parser.parse()
```

The filename in this case be accessed like this: config["filename"]. 
As we see above you can also create sub classes of configs like "tmgr" which can be used like config["tmgr"]["trainRatio"]


## Implementing Data Interface
The data interface hosts the data Module and processors. DataModule is the orchestrator and each of the data processor is used implement a stage in the data preparation stage.
    
### DataModule 
The data module is the one which calls the processor's process function. There are 2 options

**process_data(*args)** : This function accepts arguments which will be passed on to the processor, this is made for 1 node and 1 process at a time.

**multi_process_data(*args,process_count)** : This function is similar to process_data accepts the args atleast one of which should be of type list which needs to be used for multiprocessing.  multi_process_data can be run as is with multiple nodes with just process_count set if its being run in AML.

For non-AML scenarios with multi-node setting class DistributedPreprocessArguments needs to be initialized with relevant arguments

In the NER scenario we have 2 data processors 1 for reading the files and processing sentences.
The second data processor to create featurized examples from the sentences which have been read.

Data Module:

```python
class NER_dataModule(data_interface.DataInterface):
    def __init__(self) -> None:
        super(NER_dataModule, self).__init__()

    def set_datasets(self, features):
        self.train_dataset = BaseDataset(features["train"] ,isLabel = True)
        self.val_dataset = BaseDataset(features["val"] ,isLabel = True)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):   
        return self.val_dataset

    def get_test_dataset(self):
```

### Single process and Multi process dataProcessor
The switch between single and multi process is seemless. 

data.py is a single process implementation whereas data_multi.py is the multi-process counterpart.

Single process: 
```python
#create and run processor1
example_processor = Processor1(config["filename"], config["trainRatio"])
out = dataModule.process_data(example_processor)
```
Multi process:
```python
#create and run processor2
feature_processor = Processor2(config["tokenizer_type"], config["max_seq_length"], config["pad_label_id"], out)
out2 = dataModule.multi_process_data(feature_processor, out["train"], process_count=10)
```

Note: The only major restriction for multi_process is for atleast one of the arguments to be a list. A list of items/files/parameters which needs to be split to processes. Each process will call the data processor logic with a instance from the list.

## Train module and callback functions
The train module exposes callback functions where the scenario can add logic they would want to implement.
```python
class NERTrainModule(module_interface.ModuleInterface):
    def __init__(
        self,
        data: NER_dataModule,
        args: TrainModuleArgs
    ):
        super().__init__()


    def reset(self):


    def get_optimizers_schedulers(self, estimated_global_steps_per_epoch: int, epochs:int):


    def get_train_dataloader(self, sampler: torch.utils.data.Sampler, batch_size: int):


    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):


    def train_step(self, global_step: int, batch, device):


    def val_step(self, global_step: int, batch, device):

    
    def on_end_train_epoch(self, global_step:int, *train_step_collated_outputs):
        self.eval_loss, self.eval_accuracy = 0, 0
        self.predictions , self.true_labels = np.array([]), np.array([])
        self.mask = np.array([])
    
    def on_end_val_epoch(self, global_step, collated_loss, key="default"):
        active = self.mask == 1
        preds = self.predictions[active]
        labs = self.true_labels[active]
        print("Validation F1-Score: {}".format(f1_score(preds.tolist(), labs.tolist(), average='macro')))
        print("Validation Accuracy: {}".format(accuracy_score(preds.tolist(), labs.tolist())))
```

In NER we add logic for train_step, val_step where we call the model and save the output back per step/batch. We also add logic for on_end_train_epoch, on_end_val_epoch which will be executed once the train/val epoch ends.

For example above you can see the logic to calculate metrics at the end of validation epoch. and saving running loss during training in on_end_train_epoch.

