"""marlin core library"""
from marlin.core.trainer import (
    TrainerArguments,
    Trainer,
)
from marlin.core.data_interface import (
    DataProcessor,
    DataInterface,
)
from marlin.core.module_interface import (
    CallbackInterface,
    ModuleInterface,
)
from marlin.core.trainer_backend import (
    SingleProcess,
    SingleProcessAmp,
    SingleProcessApexAmp,
    DDPTrainerBackend
)

from marlin.utils.checkpointer.checkpoint_utils import (
    DefaultCheckpointerArguments,
    DefaultCheckpointer,
)
from marlin.utils.config_parser.custom_arg_parser import CustomArgParser
from marlin.utils.stats.basic_stats import BasicStats


def hello():
    '''Function docstring.'''
    print('Hello World')
