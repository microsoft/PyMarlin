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
    train_dir: None
    val_dir: None
    labels_list: None
    max_seq_len: 128
    pad_label_id: -100
    has_labels: True
    label_all_tokens: False
    tokenizer: "bert-base-uncased"
    file_format: str = "tsv"


class NERProcessor(data_interface.DataProcessor):
    """Reads a file (tsv/csv) line by line and tokenizes text using Huggingface AutoTokenizer.
       Requires header "Sentence" and "Slot" for the text and corresponding labels

    Args:
        args (arguments.DataArguments): Dataclass

    Returns:
        features (List[transformers.InputFeatures]): List of tokenized features.
    """

    def __init__(self, args):
        self.args = args
        self.label_map = (
            {label: i for i, label in enumerate(self.args.labels_list)}
            if self.args.labels_list is not None
            else None
        )
        logger.info(f"Labels map = {self.label_map}")

    def process(self, input_filepath, tokenizer):
        logger.debug(f"{multiprocessing.current_process().name} : {input_filepath}")
        if self.args.file_format == "tsv":
            sep = "\t"
        else:
            sep = ","
        df = pd.read_csv(input_filepath, sep=sep).dropna()
        examples = self._create_examples(df, tokenizer)
        return examples

    def _create_examples(self, df, tokenizer):
        """Creates examples for the training and dev sets.
        {'Sentence': 'who is harry',
        'Slot': 'O O B-contact_name'},
        """
        features = []
        for i, record in enumerate(df.to_dict("records")):
            text_a = record["Sentence"].split(" ")
            label = record["Slot"].split(" ")
            assert len(text_a) == len(label)
            features.append(
                self._convert_to_feature(
                    InputExample(guid=i, text_a=text_a, label=label), tokenizer
                )
            )
        return features

    def _convert_to_feature(self, example, tokenizer):
        tokenized_inputs = tokenizer(
            example.text_a,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True,
            max_length=self.args.max_seq_len,
        )

        label_ids = []
        if self.args.has_labels:
            word_ids = tokenized_inputs.word_ids()
            prev_word_idx = None  # To track subwords
            for word_idx in word_ids:
                if word_idx is None:  # special tokens have None
                    label_ids.append(self.args.pad_label_id)
                elif (
                    word_idx != prev_word_idx
                ):  # First part of a word always gets the label
                    label_ids.append(self.label_map[example.label[word_idx]])
                else:  # other subword tokens get the same label or ignore index, controlled by flag label_all_tokens
                    label_ids.append(
                        self.label_map[example.label[word_idx]]
                        if self.args.label_all_tokens
                        else self.args.pad_label_id
                    )
                prev_word_idx = word_idx

        feature = InputFeatures(
            input_ids=tokenized_inputs.input_ids.squeeze().tolist(),
            attention_mask=tokenized_inputs.attention_mask.squeeze().tolist(),
            token_type_ids=tokenized_inputs.token_type_ids.squeeze().tolist(),
            label=label_ids,
        )

        return feature

    def analyze(self, features):
        logger.info(f"# of features processed = {len(features)}")


class NERBaseDataset(Dataset):
    def __init__(self, features, isLabel=True):
        self.features = features
        self.isLabel = isLabel

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self._get_tensors(self.features[idx])

    def _get_tensors(self, feature):
        tensors = self._create_tensors(feature)
        if not self.isLabel:
            return tensors  # No labels
        else:
            # pylint-disable: not-callable
            label_id = torch.tensor(feature.label, dtype=torch.long) 
            tensors.update({"labels": label_id})
            return tensors

    # pylint-disable: not-callable
    def _create_tensors(self, feature):
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long) 
        input_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        tensor_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
        }
        return tensor_dict

class NERDataInterface(data_interface.DataInterface):
    '''NER Data Interface'''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = []
        self.val_dataset = []

    def setup_datasets(self, train_features, val_features, test_features=None):
        self.train_dataset = NERBaseDataset(train_features, self.args.has_labels)
        self.val_dataset = NERBaseDataset(val_features, self.args.has_labels)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_labels(self):
        return self.args.labels_list
