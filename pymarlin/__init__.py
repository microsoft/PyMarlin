"""A lightweight library for Deep Learning model training"""

__version__ = '0.2.6'
from pymarlin.core.trainer import (
    TrainerArguments,
    Trainer,
)
from pymarlin.core.data_interface import (
    DataProcessor,
    DataInterface,
)
from pymarlin.core.module_interface import (
    CallbackInterface,
    ModuleInterface,
)
from pymarlin.core.trainer_backend import (
    SingleProcess,
    SingleProcessAmp,
    SingleProcessApexAmp,
    DDPTrainerBackend,
)

from pymarlin.utils.checkpointer.checkpoint_utils import (
    DefaultCheckpointerArguments,
    DefaultCheckpointer,
)
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.stats.basic_stats import BasicStats
