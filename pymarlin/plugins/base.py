""" Base class for all plugins. """
from abc import abstractmethod
from typing import Optional, Dict
from pymarlin.core import module_interface, data_interface
from pymarlin.core import trainer as trn
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser


class Plugin:
    """Base class for all plugins.

    It is structured around three core components
    [trn.Trainer, module_interface.ModuleInterface, data_interface.DataInterface].
    Derived classes should implement the methods `setup_data`,
    `setup_module`, and `setup`. These methods will execute the data processing
    pipeline and initialize the required components for training such as
    `trainer` and `module_interface`. `setup_trainer` initializes the PyMarlin
    trainer and backend.

    `plugin.setup` is provided to bootstrap the entire pipeline for a specific
    downstream task.
    Example::

         trainer = plugin.setup()
         trainer.train()
         trainer.validate()
    """

    def __init__(self, config: Optional[Dict] = None):
        """CustomArgParser parses YAML config located at cmdline --config_path. If --config_path
        is not provided, assumes YAML file is named config.yaml and present in working directory.
            self.trainer_args (trn.TrainerArguments): Instantiated dataclass containing
        args required to initialize trn.Trainer class.
        """
        if config is None:
            config = CustomArgParser().parse()
        self.trainer_args = trn.TrainerArguments(
            **config["trainer"],
            stats_args=trn.stats.StatInitArguments(**config["stats"]),
            writer_args=trn.WriterInitArguments(**config["wrts"]),
            checkpointer_args=trn.DefaultCheckpointerArguments(**config["ckpt"])
        )

    @property
    def datainterface(self):
        """DataInterface object used for data processing.
        The property can be set in `setup_datainterface`.

        Returns:
            An object of type data_interface.DataInterface.
        """
        return self._datainterface

    @datainterface.setter
    def datainterface(self, data_interface_obj: data_interface.DataInterface):
        assert isinstance(data_interface_obj, data_interface.DataInterface)
        self._datainterface = data_interface_obj

    @property
    def dataprocessor(self):
        """DataProcessor object(s) used for data processing.
        The property may be used in conjuction with `datainterface` in the
        `setup_datainterface` method.

        Returns:
            An object of type data_interface.DataProcessor.
        """
        return self._dataprocessor

    @dataprocessor.setter
    def dataprocessor(self, data_processor_obj: data_interface.DataProcessor):
        assert isinstance(data_processor_obj, data_interface.DataProcessor)
        self._dataprocessor = data_processor_obj

    @property
    def moduleinterface(self):
        """ModuleInterface object.
        The property can be set in `setup_module`.

        Returns:
            An object of type module_interface.ModuleInterface.
        """
        return self._moduleinterface

    @moduleinterface.setter
    def moduleinterface(self, module_interface_obj: module_interface.ModuleInterface):
        assert isinstance(module_interface_obj, module_interface.ModuleInterface)
        self._moduleinterface = module_interface_obj

    @property
    def trainer(self):
        """Trainer object.
        The property can be set in `setup_trainer`.

        Returns:
            An object of type trn.Trainer.
        """
        return self._trainer

    @trainer.setter
    def trainer(self, trainer_obj: trn.Trainer):
        assert isinstance(trainer_obj, trn.Trainer)
        self._trainer = trainer_obj

    @abstractmethod
    def setup_datainterface(self, *args: Optional):
        """Derived plugins must implement this method. The method should
        execute a generic data processing pipeline for the task and update the
        TaskDataInterface object to contain the processed train and val datasets.

        NOTE to TaskPlugin designers: Typically, the plugin shouldn't need
        any input arguments from user except from the YAML config. DataInterface and
        DataProcessor related arguments should be processed in the __init__ method of
        the TaskPlugin.

        Returns:
            datainterface_obj (data_interface.DataInterface): TaskDataInterface object
        """

    @abstractmethod
    def setup_module(self, *args: Optional):
        """Derived plugins must implement this method. The method should
        create a TaskModuleInterface object (module_interface.ModuleInterface)
        and set `moduleinterface` property.

        NOTE to TaskPlugin designers: Typically, the plugin shouldn't need
        any input arguments from user. ModuleInterface related arguments should be
        processed in the __init__ method of the TaskPlugin.
        """

    def setup_trainer(self):
        """Creates a trn.Trainer object and sets the `trainer` property.
        Used by all plugins unless overriden (not recommended).
        """
        self.trainer = trn.Trainer(args=self.trainer_args, module=self.moduleinterface)

    @abstractmethod
    def setup(self, **kwargs):
        """Executes all steps from data processing to trainer initialization.

        This should be equivalent to::

             plugin.setup_datainterface()
             plugin.setup_module()
             plugin.setup_trainer()
        """
