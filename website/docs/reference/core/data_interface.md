---
sidebar_label: data_interface
title: core.data_interface
---

Data Interface module.

This module contains the abstract classes DataProcessor and
DataInterface for data processing prior to model training. Users
should implement DataProcessor&#x27;s such that instances fully
encapsulate a data processing task, and expose the process()
method for calling upon the task from a DataInterface instance,
which acts as ochestrator and implements an interface
for other modules to request train, test, validation datasets.
Please consider using the DataInterface.setup_datasets() method
to store these datasets within the DataInterface instance, and
thus have the get_[train,test,val]_dataset() methods be as
quick and computationally inexpensive as possible, as they are
called at every epoch.

## DataProcessor Objects

```python
class DataProcessor(ABC)
```

Processes and optionally analyzes data.

Designed to be used in conjunction with a DataInterface, must be extended
to implement the process() method.

#### process

```python
 | @abstractmethod
 | process(*args) -> Any
```

Process data with operations such as loading from a source, parsing,
formatting, filtering, or any required before Dataset creation.

#### analyze

```python
 | analyze(*args) -> Any
```

Optional method for analyzing data.

## DataInterface Objects

```python
class DataInterface(ABC)
```

Organizer and orchestrator for loading and processing data.

Designed to be used in conjunction with DataProcessors.
Abstract methods get_train_dataset(), get_test_dataset() and
get_val_dataset() must be implemented to return datasets.

#### \_\_init\_\_

```python
 | __init__(distrib_args: DistributedPreprocessArguments = None)
```

Accepts DistributedPreprocessArguments for custom multiprocess
rank handling.

#### setup\_datasets

```python
 | setup_datasets() -> None
```

Setup the datasets before training.

#### process\_data

```python
 | process_data(data_processor: DataProcessor, *args) -> Any
```

Process data via a DataProcessor&#x27;s process() method.

**Arguments**:

- `data_processor` _DataProcessor_ - DataProcessor to call.
  

**Returns**:

  Result of process() call.

#### multi\_process\_data

```python
 | multi_process_data(data_processor: DataProcessor, *args, *, process_count=1) -> List
```

Process data, naive multiprocessing using python multiprocessing.

Calls upon DataProcessor&#x27;s process() method with any *args provided.
All lists within args are split across processes as specified by
process_count and executed either sync or async. Any non-list
args are sent directly to the process() call. Users are encouraged
to only pass as args the objects they wish to divide among processes.

**Arguments**:

- `data_processor` _DataProcessor_ - DataProcessor to call.
- `*args` - Arguments to be passed on to DataProcessors&#x27;s process()
  method.
- `process_count` _int, optional_ - Number of worker processes to create in pool.
  

**Returns**:

- `List` - list of results to process() call per worker process.

#### get\_train\_dataset

```python
 | @abstractmethod
 | get_train_dataset(*args, **kwargs) -> Dataset
```

Returns Dataset for train data.

#### get\_val\_dataset

```python
 | @abstractmethod
 | get_val_dataset(*args, **kwargs) -> Dataset
```

Returns Dataset for val data.

#### get\_test\_dataset

```python
 | get_test_dataset(*args, **kwargs) -> Dataset
```

Returns Dataset for test data.

