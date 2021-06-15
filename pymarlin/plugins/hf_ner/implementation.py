from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from transformers import InputExample, AutoTokenizer
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser

from .data_classes import DataArguments, NERBaseDataset, NERDataInterface
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

             trainer = plugin.setup_trainer()
             trainer.train()
             trainer.validate()
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
        self.datainterface.setup_datasets()
        module_class = NERModule

        module_params = [self.module_args, self.datainterface]

        if self.distill_args.enable:
            module_params = [self.distill_args] + module_params
            module_class = build_distill_module(module_class)

        self.moduleinterface = module_class(*module_params)
