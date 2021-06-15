import dataclasses
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import InputFeatures

from pymarlin.core import data_interface
from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__, "DEBUG")


@dataclasses.dataclass
class DataArguments:
    train_filepath: str = None
    val_filepath: str = None
    labels_list: list = None
    file_format: str = "tsv"
    header: int = None
    text_a_col: int or str = None
    text_b_col: int or str = None
    label_col: int or str = None

class HfSeqClassificationDataset(Dataset):
    """PyTorch Dataset."""

    def __init__(self, args, input_filepath, label_map):
        """
        Args:
            args: DataInterface arguments
            input_filepath (str): Path to dataset
            label_map (dict): Map categorical values to numerical
        """
        self.args = args
        self.label_map = label_map
        if self.args.file_format == "json":
            self.df = pd.read_json(input_filepath, lines=True)
        elif self.args.file_format in ["tsv", "csv"]:
            if self.args.file_format == "tsv":
                sep = "\t"
            else:
                sep = ","
            self.df = pd.read_csv(input_filepath, sep=sep, header=self.args.header)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        if self.label_map is not None:
            label = self.label_map[record[self.args.label_col]]
        else:
            label = float(record[self.args.label_col])

        if self.args.text_b_col is not None:
            return record[self.args.text_a_col], record[self.args.text_b_col], label
        else:
            return record[self.args.text_a_col], label

class HfSeqClassificationDataInterface(data_interface.DataInterface):
    """Retrieves train and val PyTorch Datasets."""

    def __init__(self, args):
        """
        Args:
            args (arguments.DataArguments): Dataclass
        """
        super().__init__()
        self.args = args
        self.train_dataset = []
        self.val_dataset = []
        self._set_args()

    def _set_args(self):
        if self.args.file_format in ["tsv", "csv"]:
            if self.args.file_format == "tsv":
                sep = "\t"
            else:
                sep = ","
            if self.args.header is None:  # Refer by column numbers
                self.args.text_a_col = int(self.args.text_a_col)
                if self.args.text_b_col:
                    self.args.text_b_col = int(self.args.text_b_col)
                self.args.label_col = int(self.args.label_col)
        self.label_map = (
            {label: i for i, label in enumerate(self.args.labels_list)}
            if len(self.args.labels_list) > 1
            else None
        )

    def setup_datasets(self):
        if self.args.train_filepath is not None:
            self.train_dataset = HfSeqClassificationDataset(self.args, self.args.train_filepath, self.label_map)
        if self.args.val_filepath is not None:
            self.val_dataset = HfSeqClassificationDataset(self.args, self.args.val_filepath, self.label_map)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_labels(self):
        return self.args.labels_list
