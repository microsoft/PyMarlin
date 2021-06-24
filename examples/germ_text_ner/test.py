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

#### Cmdline: python test.py --data.train_filepath ./train_germ/train.tsv --data.val_filepath ./val_germ/dev.tsv --config_path config_germ.yaml

    plugin.setup_trainer()
    trainer = plugin.trainer
    trainer.train()
