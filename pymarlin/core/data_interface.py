# -*- coding: utf-8 -*-
"""
Data Interface module:

This module contains the abstract classes DataProcessor and
DataInterface for data processing prior to model training. Users
should implement DataProcessor's such that instances fully
encapsulate a data processing task, and expose the process()
method for calling upon the task from a DataInterface instance,
which acts as ochestrator and implements an interface
for other modules to request train, validation datasets.
Please consider using the DataInterface.setup_datasets() method
to store these datasets within the DataInterface instance, and
thus have the get_[train,val]_dataset() methods be as
quick and computationally inexpensive as possible, as they are
called at every epoch.
"""

import multiprocessing
import itertools
from typing import Any, List
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from pymarlin.utils.distributed import DistributedPreprocessArguments
from pymarlin.utils import fabrics
from pymarlin.utils import distributed as dist


class DataProcessor(ABC):
    """
    Processes and optionally analyzes data.

    Designed to be used in conjunction with a DataInterface, must be extended
    to implement the process() method.
    """

    def __init__(self, distrib_args: DistributedPreprocessArguments = None):
        """
        Accepts DistributedPreprocessArguments for custom multiprocess
        rank handling.
        """
        self.distrib_args = distrib_args

    @abstractmethod
    def process(self, *args) -> Any:
        """
        Process data with operations such as loading from a source, parsing,
        formatting, filtering, or any required before Dataset creation.
        """

    def analyze(self) -> Any:
        """
        Optional method for analyzing data.
        """

    def process_data(self, *args) -> Any:
        """
        Process data via a DataProcessor's process() method.

        Args:
            data_processor (DataProcessor): DataProcessor to call.

        Returns:
            Result of process() call.
        """
        res_process = self.process(*args)
        self.analyze()
        return res_process

    def multi_process_data(self,
                           *args,
                           process_count=1) -> List:
        """
        Process data, naive multiprocessing using python multiprocessing.

        Calls upon DataProcessor's process() method with any *args provided.
        All lists within args are split across processes as specified by
        process_count and executed either sync or async. Any non-list
        args are sent directly to the process() call. Users are encouraged
        to only pass as args the objects they wish to divide among processes.

        Args:
            data_processor (DataProcessor): DataProcessor to call.
            *args: Arguments to be passed on to DataProcessors's process()
                method.
            process_count (int, optional): Number of worker processes to create in pool.

        Returns:
            List: list of results to process() call per worker process.
        """
        self._set_ranks()
        list_params = self._collect_params(*args)

        with multiprocessing.Pool(processes=process_count) as p:
            res_process_list = p.starmap(self.process,
                                         zip(*list_params))
            p.close()
            p.join()
        if any(res_process_list):
            res_process_list = list(itertools.chain.from_iterable(res_process_list))
        self.analyze()
        return res_process_list

    def _collect_params(self, *args) -> List:
        """
        Auxiliary function for multi_process_data().

        Transforms arguments into lists of length equal to the longest
        argument of type list.

        Args:
            *args: Arguments to be converted to type list.

        Returns:
            List: list of arguments converted to type list.
        """
        try:
            max_length = max([len(self._get_node_params(param))
                              for param in args
                              if isinstance(param, list)])
        except ValueError as err:
            raise ValueError("At least one argument must be of type"
                             "list when calling multi_process_data.") from err

        list_params = []
        for param in args:
            if isinstance(param, list):
                # Parameters of type list of len > 1 are split across the process
                if len(param) > 1:
                    list_params.append(self._get_node_params(param))
                # Parameters of len = 1 are duplicated max_length times
                else:
                    list_params.append(max_length * param)
            # Non-list params are copied and passed to each process instance
            else:
                list_params.append(max_length * [param])
        return list_params

    def _set_ranks(self):
        """
        Set ranks used for distributed data pre-processing.

        This method will attempt to infer the distributed training arguments
        from environment variables in the case that they are not explicitly
        provided.
        """
        if self.distrib_args is None:
            distributed_args = DistributedPreprocessArguments()
            if fabrics.is_azureml_mpirun():
                distributed_args = dist.fetch_ranks_from_azureml_preprocess()
            self.distrib_args = distributed_args

    def _get_node_params(self, param):
        """
        Creates a subset of parameters to be processed by the current node.

        Args:
            param (list): List of parameters to be split among nodes for
                processing.

        Returns:
            Subset of provided parameters to be processed by current node.
        """
        node_params = []

        if len(param) < self.distrib_args.node_count:
            n_step = 1
        else:
            n_step = len(param) // self.distrib_args.node_count
        if not self.distrib_args.node_rank:
            self.distrib_args.node_rank = self.distrib_args.global_rank // \
                self.distrib_args.local_size

        n_inputs = self.distrib_args.node_rank * n_step
        while n_inputs < len(param):
            node_params.extend(param[n_inputs:n_inputs + n_step])
            n_inputs += n_step*self.distrib_args.node_count

        return node_params

class DataInterface(ABC):
    """
    Organizer and orchestrator for loading and processing data.

    Designed to be used in conjunction with DataProcessors.
    Abstract methods get_train_dataset() and
    get_val_dataset() must be implemented to return datasets.
    """

    def setup_datasets(self) -> None:
        """
        Setup the datasets before training.
        """

    @abstractmethod
    def get_train_dataset(self, *args, **kwargs) -> Dataset:
        """
        Returns Dataset for train data.
        """

    @abstractmethod
    def get_val_dataset(self, *args, **kwargs) -> Dataset:
        """
        Returns Dataset for val data.
        """
 