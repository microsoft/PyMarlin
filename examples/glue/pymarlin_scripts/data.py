import os, sys
import json
import argparse
import pickle
import csv
import dataclasses

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import InputFeatures, AutoTokenizer

from pymarlin.core.data_interface import DataProcessor, DataInterface
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__)

from init_args import DataInterfaceArguments, DataProcessorArguments
from glue_processors import glue_dict

class Featurizer(DataProcessor):
    def __init__(self, args, labels_list, tokenizer=None):
        self.args = args
        self.label_map = {label: i for i, label in enumerate(labels_list)} if None not in labels_list else None
        self.tokenizer = tokenizer

    def process(self, examples, output_path, save_features=False):
        self.features = []
        for example in examples:
            tokens = self.tokenizer(example.text_a,
                                example.text_b,
                                max_length=self.args.max_seq_len,
                                padding='max_length',
                                truncation=True,
                                return_token_type_ids=True,
                                return_tensors='pt')

            if example.label is not None:
                if self.label_map is not None: # classification task
                    label = self.label_map[example.label]
                else: # regression task data processor returns labels list [None]
                    label = float(example.label)
            else: # labels not provided (only inference)
                label = None
            self.features.append(InputFeatures(tokens.input_ids.squeeze().tolist(),
                                          tokens.attention_mask.squeeze().tolist(),
                                          tokens.token_type_ids.squeeze().tolist(),
                                          label))
        if save_features:
            with open(output_path, 'wb') as f:
                pickle.dump(self.features, f)
        return self.features

    def analyze(self):
        logger.info(f"# of features processed = {len(self.features)}")


class TaskDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.load_features()

    def load_features(self):
        with open(self.datapath, 'rb') as f:
            self.features = pickle.load(f)
            
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return self._create_tensors(feature)
    
    def _create_tensors(self, feature):
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        tensor_dict = {'input_ids':input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids}
        if feature.label is not None:
            if type(feature.label) == int:
                label_id = torch.tensor(feature.label, dtype=torch.long)
            else:
                label_id = torch.tensor(feature.label, dtype=torch.float)
            tensor_dict.update({'labels': label_id})
        return tensor_dict

class TaskData(DataInterface):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def create_dataset(self, datapath):
        return TaskDataset(datapath)
        
    def get_train_dataset(self, trainpath):
        return self.create_dataset(trainpath)
        
    def get_val_dataset(self, valpath):
        return self.create_dataset(valpath)
    
    def get_test_dataset(self, testpath):
        return self.create_dataset(testpath)


if __name__ == "__main__":

    parser = CustomArgParser(log_level='DEBUG')
    config = parser.parse()
    logger.info(f"final merged config = {config}\n")

    os.makedirs(config['dmod']['output_dir'], exist_ok=True)

    dm_args = DataInterfaceArguments(**config['dmod'])
    data = TaskData(dm_args)

    proc_args = DataProcessorArguments(**config['dproc'])
    processor = glue_dict[proc_args.task.lower()]['processor'](proc_args)

    input_file = proc_args.set_type + ".tsv"
    input_filepath = os.path.join(dm_args.input_dir, input_file)
    examples = data.process_data(processor, input_filepath)

    labels_list = processor.get_labels()
    tokenizer = AutoTokenizer.from_pretrained(proc_args.tokenizer)
    featurizer = Featurizer(proc_args, labels_list, tokenizer)

    output_filepath = os.path.join(dm_args.output_dir, input_file)
    features = data.process_data(featurizer, examples, output_filepath, True)

# python data.py --dmod.input_dir "raw_data\RTE" --dmod.output_dir "processed_data\RTE\train" --dproc.task "RTE" --dproc.set_type "train"
