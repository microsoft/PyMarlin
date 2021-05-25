import os
import json
import multiprocessing
import dataclasses
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import InputFeatures

from pymarlin.utils.logger.logging_utils import getlogger

logger = getlogger(__name__, "DEBUG")

from pymarlin.core import data_interface


@dataclasses.dataclass
class DataArguments:
    train_dir: str = None
    val_dir: str = None
    labels_list: list = None
    file_format: str = "tsv"
    header: int = None
    text_a_col: int or str = None
    text_b_col: int or str = None
    label_col: int or str = None
    max_seq_len: int = 128
    has_labels: bool = True
    tokenizer_path: str = None
    hf_tokenizer: str = "bert-base-uncased"
    cpu_threads: int = -1


class HfSeqClassificationProcessor(data_interface.DataProcessor):
    """Reads a file (tsv/csv/json) line by line and tokenizes text using Huggingface AutoTokenizer.

    Args:
        args (arguments.DataArguments): Dataclass

    Returns:
        features (List[transformers.InputFeatures]): List of tokenized features.
    """

    def __init__(self, args):
        self.args = args
        self.label_map = (
            {label: i for i, label in enumerate(self.args.labels_list)}
            if len(self.args.labels_list) > 1
            else None
        )
        logger.info(f"Labels map = {self.label_map}")

    def process(self, input_filepath, tokenizer):
        if self.args.file_format == "json":
            df = pd.read_json(input_filepath, lines=True)
        elif self.args.file_format in ["tsv", "csv"]:
            if self.args.file_format == "tsv":
                sep = "\t"
            else:
                sep = ","
            df = pd.read_csv(input_filepath, sep=sep, header=self.args.header)
            if self.args.header is None:  # Refer by column numbers
                self.args.text_a_col = int(self.args.text_a_col)
                if self.args.text_b_col:
                    self.args.text_b_col = int(self.args.text_b_col)
                self.args.label_col = int(self.args.label_col)
        features = []
        for i, record in enumerate(df.to_dict("records")):
            text_a = record[self.args.text_a_col]
            text_b = (
                record[self.args.text_b_col]
                if self.args.text_b_col is not None
                else None
            )
            label = (
                record[self.args.label_col]
                if (self.args.has_labels and self.args.label_col is not None)
                else None
            )

            # skip_row_0 = False
            # if self.args.file_format in ["tsv", "csv"]:
            #     if self.args.file_format == "tsv":
            #         sep = "\t"
            #     else:
            #         sep = ","
            #     if self.args.header:
            #         skip_row_0 = True
            #     self.args.text_a_col = int(self.args.text_a_col)
            #     if self.args.text_b_col:
            #         self.args.text_b_col = int(self.args.text_b_col)
            #     self.args.label_col = int(self.args.label_col)

            # features = []
            # with open(input_filepath, 'r', encoding='utf-8') as f:
            #     for i, line in enumerate(f):
            #         if skip_row_0 and i == 0:
            #             logger.info(f"{multiprocessing.current_process().name} : Skipping header row 0")
            #             continue
            #         if self.args.file_format == "json":
            #             line = json.loads(line)
            #         else:
            #             print(line)
            #             line = line.strip().split(sep)
            #             print(line)
            #         text_a = line[self.args.text_a_col]
            #         if self.args.text_b_col:
            #             text_b = line[self.args.text_b_col]
            #         else:
            #             text_b = None
            #         if not self.args.has_labels or self.args.label_col is None:
            #             label = None
            #         else:
            #             label = line[self.args.label_col]

            tokens = tokenizer(
                text_a,
                text_b,
                max_length=self.args.max_seq_len,
                truncation=True,
                padding="max_length",
                return_token_type_ids=True,
                return_tensors="pt",
            )
            if label is not None:
                if self.label_map is not None:  # classification
                    label = self.label_map[label]
                else:  # regression doesn't have a label map
                    label = float(label)
            features.append(
                InputFeatures(
                    tokens.input_ids.squeeze().tolist(),
                    tokens.attention_mask.squeeze().tolist(),
                    tokens.token_type_ids.squeeze().tolist(),
                    label,
                )
            )
        logger.debug(
            f"{multiprocessing.current_process().name} : {len(features)} features created"
        )
        return features

    def analyze(self, features):
        logger.info(f"# of features processed = {len(features)}")


class HfSeqClassificationDataset(Dataset):
    """PyTorch Dataset."""

    def __init__(self, features):
        """
        Args:
            features (List[transformers.InputFeatures]): List of tokenized features.
        """
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return self._create_tensors(feature)

    def _create_tensors(self, feature):
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        tensor_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
        }
        if feature.label is not None:
            if isinstance(feature.label, int):
                label_id = torch.tensor(feature.label, dtype=torch.long)
            else:
                label_id = torch.tensor(feature.label, dtype=torch.float)
            tensor_dict.update({"labels": label_id})
        return tensor_dict


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

    def setup_datasets(self, train_features, val_features):
        self.train_dataset = HfSeqClassificationDataset(train_features)
        self.val_dataset = HfSeqClassificationDataset(val_features)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_labels(self):
        return self.args.labels_list
