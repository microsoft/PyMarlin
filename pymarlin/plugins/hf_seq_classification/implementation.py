from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.core import data_interface, module_interface
from pymarlin.plugins.base import Plugin
from pymarlin.plugins.hfdistill_utils import build_distill_module, DistillationArguments

from .data_classes import (
    HfSeqClassificationDataInterface,
    DataArguments,
)
from .module_classes import (
    HfSeqClassificationModule,
    ModuleInterfaceArguments,
    ModelArguments,
)
from typing import Optional, Dict


class HfSeqClassificationPlugin(Plugin):
    """Plugin for Text Sequence Classification using Huggingface models.


    plugin.setup() bootstraps the entire pipeline and returns a fully setup trainer.
    Example::

             trainer = plugin.setup()
             trainer.train()
             trainer.validate()

    Alternatively, you can run `setup_datainterface` `setup_module` `setup_trainer` individually.
    Example::

             plugin.setup_datainterface()
             plugin.setup_module()
             trainer = plugin.setup_trainer()
    """

    def __init__(self, config: Optional[Dict] = None):
        """CustomArgParser parses YAML config located at cmdline --config_path. If --config_path
        is not provided, assumes YAML file is named config.yaml and present in working directory.
        Instantiates dataclasses:
            self.data_args (arguments.DataInterfaceArguments): Instantiated dataclass containing
            args.
            self.module_args (arguments.ModuleInterfaceArguments): Instantiated dataclass containing
            args required to initialize HfSeqClassificationModule class.
            self.distill_args (arguments.DistillationArguments): Instantiated dataclass
            required to initialize DistillHfModule.
                Set self.distill_args.enable = True in config file to do knowledge distillation
                instead of regular training.
        Sets properties:
            self.datainterface: data_interface.DataInterface [HfSeqClassificationDataInterface] object
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

    def setup_datainterface(self):
        """Calls `datainterface.setup_datasets(train_data, val_data)`.

        Assumptions:
            Training and validation files are placed in separate directories.
            Accepted file formats: source/target text lines in data_args.data_dir/{train,val}.{source,targets}
        """
        self.datainterface = HfSeqClassificationDataInterface(self.data_args)
        self.datainterface.setup_datasets()

    def setup_module(self):
        """Sets `HfSeqClassificationModule.data` property to `datainterface` which contains
        the processed datasets. Assertion error is thrown if `datainterface` retrieves no train
        or val data, indicating that `datainterface` hasn't been setup with processed data.
        Sets the `HfSeqClassificationModule.model` property after initializing weights:
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
        module_class = HfSeqClassificationModule
        module_params = [self.module_args, self.datainterface]
        if self.distill_args.enable:
            module_params = [self.distill_args] + module_params
            module_class = build_distill_module(module_class)
        self.moduleinterface = module_class(*module_params)

    def setup(self):
        """Executes all the setup methods required to create a trn.Trainer object.
        Trainer needs `moduleinterface` and backend is specified by self.trainer_args.backend.
        """
        self.setup_datainterface()
        self.setup_module()
        self.setup_trainer()