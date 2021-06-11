import multiprocessing
import os

import itertools
import pandas as pd
from pymarlin.core.trainer_backend import build_trainer_backend

from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__, 'DEBUG')
from pymarlin.core import data_interface
from transformers import InputExample, AutoTokenizer, InputFeatures
from pymarlin.plugins import HfNERPlugin

if __name__ == '__main__':
    ########### Usage #############
    plugin = HfNERPlugin()

################ Run plugin.setup() to bootstrap entire pipeline #################

#### Cmdline: python test.py --config_path "config.yaml" --data.train_dir <Path to train dir> --data.val_dir <Path to train dir>

    plugin.setup()
    plugin.trainer.train()
    plugin.trainer.validate()

