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

    Example:
    ```python
    trainer = plugin.setup_trainer()
    trainer.train()
    trainer.validate()
    ```
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

        self.datainterface = HfSeqClassificationDataInterface(self.data_args)
        self.datainterface.setup_datasets()
        module_class = HfSeqClassificationModule
        module_params = [self.module_args, self.datainterface]
        if self.distill_args.enable:
            module_params = [self.distill_args] + module_params
            module_class = build_distill_module(module_class)
        self.moduleinterface = module_class(*module_params)
