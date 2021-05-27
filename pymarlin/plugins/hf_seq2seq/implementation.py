import os
import multiprocessing

from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.logger.logging_utils import getlogger

logger = getlogger(__name__, "DEBUG")

from pymarlin.core import data_interface, module_interface
from pymarlin.core import trainer as trn

from pymarlin.plugins.base import Plugin
from .data_classes import HfSeq2SeqData, DataInterfaceArguments
from .module_classes import (
    HfSeq2SeqModule,
    ModuleInterfaceArguments,
    ModelArguments,
    GenerateArguments,
)


class HfSeq2SeqPlugin(Plugin):
    """Plugin for Text Sequence to Sequence Generation using Huggingface models.

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

    def __init__(self, config=None):
        """Accepts optional config dictionary.
        CustomArgParser parses YAML config located at cmdline --config_path. If --config_path
        is not provided, assumes YAML file is named config.yaml and present in working directory.
        Instantiates dataclasses:
            self.data_args (arguments.DataInterfaceArguments): Data Inference arguments
            self.module_args (arguments.ModuleInterfaceArguments): Module Interface Arguments
        Sets properties:
            self.datainterface: data_interface.DataInterface [HfSeq2SeqData] object
            self.moduleinterface: module_interface.ModuleInterface [HfSeq2SeqModule] object
        """
        super().__init__()
        if config is None:
            config = CustomArgParser(log_level="DEBUG").parse()
        self.data_args = DataInterfaceArguments(**config["data"])
        self.module_args = ModuleInterfaceArguments(
            **config["module"],
            model_args=ModelArguments(**config["model"]),
            generate_args=GenerateArguments(**config["generate"])
        )
        # self.distill_args = DistillationArguments(**config['distill'])

    def setup_datainterface(self):
        """Calls `datainterface.setup_datasets(train_data, val_data)`.

        Assumptions:
            Training and validation files are placed in separate directories.
            Accepted file formats: source/target text lines in data_args.data_dir/{train,val}.{source,targets}
        """
        self.datainterface = HfSeq2SeqData(self.data_args)
        self.datainterface.setup_datasets()

    def setup_module(self):
        """Sets `HfSeq2SeqModule.data` property to `datainterface` which contains
        the processed datasets. Assertion error is thrown if `datainterface` retrieves no train
        or val data, indicating that `datainterface` hasn't been setup with processed data.
        Sets the `HfSeq2SeqModule.model` property after initializing weights:
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
        # datainterface should contain the processed datasets
        assert (
            len(self.datainterface.get_train_dataset()) != 0
            or len(self.datainterface.get_val_dataset()) != 0
        )
        self.moduleinterface = HfSeq2SeqModule(self.datainterface, self.module_args)

    def setup(self):
        """Executes all the setup methods required to create a trn.Trainer object.
        Trainer needs `moduleinterface` and backend is specified by self.trainer_args.backend.
        """
        self.setup_datainterface()
        self.setup_module()
        self.setup_trainer()
