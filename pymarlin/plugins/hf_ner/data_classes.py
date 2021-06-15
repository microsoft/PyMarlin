'''pymarlin.pluguins.hf_ner.data_classes'''
import dataclasses
import pandas as pd
import multiprocessing
import torch

from torch.utils.data import Dataset
from transformers import InputExample, InputFeatures
from pymarlin.utils.logger.logging_utils import getlogger
from pymarlin.core import data_interface

logger = getlogger(__name__, "DEBUG")

@dataclasses.dataclass
class DataArguments:
    train_filepath: None
    val_filepath: None
    labels_list: None
    has_labels: True
    file_format: str = "tsv"


class NERBaseDataset(Dataset):
    def __init__(self, args, input_filepath):
        self.input_filepath = input_filepath
        self.args = args

        if self.args.file_format == "tsv":
            sep = "\t"
        else:
            sep = ","
        self.df = pd.read_csv(self.input_filepath, sep=sep).dropna()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sent = record["Sentence"].split(" ")
        label = record["Slot"].split(" ")
        assert len(sent) == len(label)
        return sent,label 

class NERDataInterface(data_interface.DataInterface):
    '''NER Data Interface'''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = []
        self.val_dataset = []
        self._set_args()

    def setup_datasets(self):
        self.train_dataset = NERBaseDataset(self.args, self.args.train_filepath)
        self.val_dataset = NERBaseDataset(self.args, self.args.val_filepath)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_labels(self):
        return self.args.labels_list

    def _set_args(self):
        self.label_map = (
            {label: i for i, label in enumerate(self.args.labels_list)}
            if self.args.labels_list is not None
            else None
        )
        logger.info(f"Labels map = {self.label_map}")
