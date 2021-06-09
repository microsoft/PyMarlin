import os
import sys
import json
import csv
import multiprocessing
import itertools
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from transformers import InputExample, AutoTokenizer
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser

from .data_classes import DataArguments, NERProcessor, NERBaseDataset, NERDataInterface
from .module_classes import NERModule, ModuleInterfaceArguments, ModelArguments

from transformers import AutoModelForTokenClassification
from pymarlin.core.data_interface import DataProcessor, DataInterface
from pymarlin.core.module_interface import ModuleInterface
from pymarlin.plugins.base import Plugin
from pymarlin.plugins.hfdistill_utils import build_distill_module, DistillationArguments
from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger.logging_utils import getlogger

logger = getlogger(__name__, "DEBUG")


class HfNERPlugin(Plugin):
    """Named Entity Recognition or Token Classification plugin for HuggingFace models

    plugin.setup() bootstraps the entire pipeline and returns a fully setup trainer.
    Example:

             trainer = plugin.setup()
             trainer.train()
             trainer.validate()

    Alternatively, you can run `setup_datainterface` `setup_module` `setup_trainer` individually.
    Example:

             plugin.setup_datainterface()
             plugin.setup_module()
             trainer = plugin.setup_trainer()
    """

    def __init__(self, config: Optional[Dict] = None):
        """CustomArgParser parses YAML config located at cmdline --config_path. If --config_path
        is not provided, assumes YAML file is named config.yaml and present in working directory.
        Instantiates dataclasses:
            self.data_args (arguments.DataInterfaceArguments): Instantiated dataclass containing
            args required to initialize NERDataInterface and NERProcessor classes
            self.module_args (arguments.ModuleInterfaceArguments): Instantiated dataclass containing
            args required to initialize NERModule class

        Sets properties:
            self.datainterface: data_interface.DataInterface [NERDataInterface] object
            self.dataprocessor: data_interface.DataProcessor [NERProcessor] object.
                These two together are used to read raw data and create sequences of tokens in `setup_datainterface`.
                The processed data is fed to HuggingFace AutoModelForTokenClassification models.
            self.module: module_interface.ModuleInterface [NERModule] object
                This is used to initialize a Marlin trainer.
        """
        super().__init__(config=None)
        if config is None:
            config = CustomArgParser(log_level="DEBUG").parse()
        self.data_args = DataArguments(**config["data"])
        self.module_args = ModuleInterfaceArguments(
            **config["module"], model_args=ModelArguments(**config["model"])
        )
        self.distill_args = DistillationArguments(**config["distill"])

        self.datainterface = NERDataInterface(self.data_args)
        self.dataprocessor = NERProcessor(self.data_args)
        module_class = NERModule

        module_params = [self.module_args]

        if self.distill_args.enable:
            module_params = [self.distill_args] + module_params
            module_class = build_distill_module(module_class)

        self.moduleinterface = module_class(*module_params)

    def setup_datainterface(self):
        """Executes the data processing pipeline. Tokenizes train and val datasets using the
        `dataprocessor` and `datainterface`.
        Finally calls `datainterface.setup_datasets(train_data, val_data)`.

        Assumptions:
            Training and validation files are placed in separate directories.
            Accepted file formats: tsv, csv.
            Format of input files 2 columns 'Sentence', 'Slot'
            Example
            {'Sentence': 'who is harry',
            'Slot': 'O O B-contact_name'},
        """
        tokenizer = AutoTokenizer.from_pretrained(self.data_args.tokenizer)

        train_features, val_features = [], []
        if self.data_args.train_dir is not None:
            train_files = [
                os.path.join(self.data_args.train_dir, filename)
                for filename in os.listdir(self.data_args.train_dir)
            ]
            train_features = self.datainterface.multi_process_data(
                self.dataprocessor,
                train_files,
                tokenizer,
                process_count=multiprocessing.cpu_count(),
            )
            logger.info(f"Train features created  {len(train_features)} ")
        if self.data_args.val_dir is not None:
            val_files = [
                os.path.join(self.data_args.val_dir, filename)
                for filename in os.listdir(self.data_args.val_dir)
            ]
            val_features = self.datainterface.multi_process_data(
                self.dataprocessor,
                val_files,
                tokenizer,
                process_count=multiprocessing.cpu_count(),
            )
            logger.info(f"Val features created  {len(val_features)} ")
        self.datainterface.setup_datasets(train_features, val_features)

    def setup_module(self, module_interface=None):
        """Sets `NERModule.data` property to `datainterface` which contains
        the processed datasets. Assertion error is thrown if `datainterface` retrieves no train
        or val data, indicating that `datainterface` hasn't been setup with processed data.
        Sets the `NERModule.model` property after initializing weights:
            Option 1: Load weights from specified files mentioned in YAML config
                        model:
                            model_config_path
                            model_config_file
                            model_path
                            model_file
            Option 2: Load from Huggingface model hub, specify string in YAML config as:
                        model:
                            hf_model
        """
        assert (
            len(self.datainterface.get_train_dataset()) != 0
            or len(self.datainterface.get_val_dataset()) != 0
        )
        self.moduleinterface.data = self.datainterface
        self.moduleinterface.setup_model(AutoModelForTokenClassification)

    def setup(self):
        """Method to be called to use plugin out of box. This method will complete preprocessing , create datasets
        setup the module interface and trainer.
        """
        self.setup_datainterface()
        self.setup_module()
        self.setup_trainer()
