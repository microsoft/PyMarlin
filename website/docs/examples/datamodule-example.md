# Data interface single and multi process

This is an example explaining how to leverage the in-built multiprocessing capability of DataInterface for large amounts of data.
For example purpose we are using 27 files from wikipedia raw text.
1) Azure virtual machine , single node multi-process , single selective machine
2) AML, single node vs multi-node, single selective machine
## Configs - YAML and Parsing

For ease of use we have configs passed in as YAML files. 
In this case we use the config file : config_prod.yaml included with example code.

Snippet of config: ( modify file paths according to your folder structure)

```python
input_dir: 'C:/Users/ashwinsr/wikipedia.part1'
out_dir: 'C:/Users/ashwinsr/out_fold'
process_count: 10
run_type: ''
```

This config can be read in like below : 

```python
#Create arg parser and read config
parser = CustomArgParser(log_level='DEBUG', default_yamlfile="config_prod.yaml")
config = parser.parse()
```

Our data processor is a simple token splitter which given raw text will split it into token store the results back in a file. The processor runs 1 file at a time.

## Virtual machine
### Single virtual machine with multi process

```python
dataInterface = Ex_dataInterface()  
file_list = dataInterface.get_file_names(config["input_dir"])
#create and run processor1
example_processor = Processor1(config["input_dir"], config["out_dir"])
out = example_processor.multi_process_data(file_list, process_count=config["process_count"])
```
Here we create a list of files in the directory and initialize the processor with the input and output directory. We call the the multi_process_data function in the processor, passing the list of files , with the process count. The processor then spins up those many number of processes to create coressponding output. 

### Selective node preprocessing
For a case where we have a single node but want to process the data in batches. We want the processor to run on different subset of files depending upon the rank we assign. This is to emulate multi-node behaviour with a single node by controlling the node rank parameter.

For instance if we have 30 files to process over 5 separate runs , then we need to add the following to config and initialize dataProcessor accordingly

```python
distribArgs:
    local_rank:  0
    global_rank:  0
    world_size:  1
    node_count:  5
    local_size:  1
    node_rank: 3
```

```python
distrib = DistributedPreprocessArguments(**config["distribArgs"])
example_processorer = Processor1(config["input_dir"], config["out_dir"], distrib)
```
Remember to initialize the base dataProcessor class with the distributed arguemnts as shown below, the default None would treat it like a regular multi-node processing job

```
class Processor1(data_interface.DataProcessor):
    def __init__(self, input_dir, out_dir, distrib_args = None):
        super(Processor1, self).__init__(distrib_args)
        self.input_dir = input_dir
        self.out_dir = out_dir
```

With the above setting we would process files 18-24 out of 30. Since the node_rank is 3 (0 indexed) and can be a maximum of 4. node_count gives us a count of total nodes available
This gives a flexibility with large data processing with limited compute.

To run in virtual machine copy over the files to virtual machine using SCP 
Install pymarlin and requirements and run example
```python
    > ssh $user@$machine -p $port
    $ pip install  ./pymarlin --force-reinstall
    $ pip install -r pymarlin/requirements.txt
    $ cd data_ex
    $ python data.py
```

### AML 
We can do single and multi-node processing both with AML. The datamodule handles AML ranking internally for both single and multinodes to appropriately divide the files across nodes. 
You will find a notebook along with the example to submit a AML a job, with placeholders for storage and compute accounts.
