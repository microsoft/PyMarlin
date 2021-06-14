from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.logger.logging_utils import getlogger

logger = getlogger(__name__, "INFO")

from pymarlin.core import data_interface, module_interface
from pymarlin.core import trainer as trn
from pymarlin.plugins.base import Plugin
from pymarlin.plugins.hfdistill_utils import build_distill_module, DistillationArguments

from .data_classes import (
    HfSeqClassificationDataInterface,
    # HfSeqClassificationProcessor,
    DataArguments,
)
from .module_classes import (
    HfSeqClassificationModule,
    ModuleInterfaceArguments,
    ModelArguments,
)

import os
from typing import Optional, Dict
import multiprocessing

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HfSeqClassificationPlugin(Plugin):
    """Plugin for Text Sequence Classification using Huggingface models.

    plugin.setup() bootstraps the entire pipeline and returns a fully setup trainer.
    Example:
    ```python
    trainer = plugin.setup()
    trainer.train()
    trainer.validate()
    ```

    Alternatively, you can run `setup_datainterface` `setup_module` `setup_trainer` individually.
    Example:
    ```python
    plugin.setup_datainterface()
    plugin.setup_module()
    trainer = plugin.setup_trainer()
    ```
    """

    def __init__(self, config: Optional[Dict] = None):
        """CustomArgParser parses YAML config located at cmdline --config_path. If --config_path
        is not provided, assumes YAML file is named config.yaml and present in working directory.
        Instantiates dataclasses:
            self.data_args (arguments.DataInterfaceArguments): Instantiated dataclass containing
            args required to initialize HfSeqClassificationDataInterface and HfSeqClassificationProcessor
            classes.
            self.module_args (arguments.ModuleInterfaceArguments): Instantiated dataclass containing
            args required to initialize HfSeqClassificationModule class.
            self.distill_args (arguments.DistillationArguments): Instantiated dataclass
            required to initialize DistillHfModule.
                Set self.distill_args.enable = True in config file to do knowledge distillation
                instead of regular training.
        Sets properties:
            self.datainterface: data_interface.DataInterface [HfSeqClassificationDataInterface] object
            self.dataprocessor: data_interface.DataProcessor [HfSeqClassificationProcessor] object.
                These two together are used to read raw data and create sequences of tokens in `setup_datainterface`.
                The processed data is fed to HuggingFace AutoModelForSequenceClassification models.
            self.module: module_interface.ModuleInterface [HfSeqClassificationModule] object
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

        self.datainterface = HfSeqClassificationDataInterface(self.data_args)
        self.datainterface.setup_datasets()
        module_class = HfSeqClassificationModule
        module_params = [self.module_args, self.datainterface]
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
            Accepted file formats: json, tsv, csv.
            YAML config file should specify the column names or ids:
                data:
                    text_a_col
                    text_b_col (optional None)
                    label_col (optional None)
            Header row is skipped for tsv/csv file if data_args.header = True
            data_args.hf_tokenizer: String corresponding to Huggingface AutoTokenizer
            data_args.cpu_threads: Number of processes to use for Python CPU multiprocessing
        """
        pass
