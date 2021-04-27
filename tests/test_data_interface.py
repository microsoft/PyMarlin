"""Test module for data_interface"""

import os
from dataclasses import dataclass
import unittest
import pandas as pd
from torch.utils.data import Dataset
from pymarlin.core.data_interface import DataInterface, DataProcessor

@dataclass
class MyArgs:
    filepath_train: str = os.path.join("outputs", "file1.csv")
    filepath_test: str = os.path.join("outputs", "file2.csv")
    text_field: str = "text"
    label_field: str = "label"


class MyDataset(Dataset):
    def __init__(self, df, text_field, label_field):
        self.df = df
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (row[self.text_field], row[self.label_field])


class MyData(DataInterface):

    def __init__(self):
        super().__init__()
        self._train_ds = None
        self._val_ds = None

    def setup_datasets(self, train_ds, val_ds):
        self._train_ds = train_ds
        self._val_ds = val_ds

    def get_train_dataset(self):
        return self._train_ds

    def get_val_dataset(self):
        return self._val_ds


class MyDataProcessor(DataProcessor):

    def __init__(self, args):
        self.args = args

    def process(self):
        df = pd.read_csv(self.args.filepath_train)
        return MyDataset(df, self.args.text_field,
                         self.args.label_field)


class MyDataMultiProcessor(DataProcessor):

    def __init__(self, args):
        self.args = args

    def process(self, filename):
        df = pd.read_csv(filename)
        return MyDataset(df, self.args.text_field,
                         self.args.label_field)


class TestDataInterface(unittest.TestCase):

    def setUp(self):

        self.args = MyArgs()
        self.data_interface = MyData()
        self.data_processor = MyDataProcessor(self.args)
        self.data_multiprocessor = MyDataMultiProcessor(self.args)

        if not os.path.exists("outputs"):
            os.makedirs("outputs")

    def tearDown(self):
        for f in os.listdir("outputs"):
            if f.endswith(".csv"):
                os.remove(os.path.join("outputs", f))

    def test_process_data(self):
        df = pd.DataFrame({self.args.text_field: ['one', 'two'],
                           self.args.label_field: [1, 2]})
        df.to_csv(self.args.filepath_train)

        train_ds = self.data_interface.process_data(self.data_processor)
        assert train_ds[0] == ('one', 1)
        assert len(train_ds) == 2

    def test_multi_process_data(self):
        df1 = pd.DataFrame({self.args.text_field: ['one', 'two'],
                            self.args.label_field: [1, 2]})
        df2 = pd.DataFrame({self.args.text_field: ['three', 'four', 'five'],
                            self.args.label_field: [3, 4, 5]})
        df1.to_csv(self.args.filepath_train)
        df2.to_csv(self.args.filepath_test)

        train_ds_list = self.data_interface.multi_process_data(
            self.data_multiprocessor,
            [self.args.filepath_train, self.args.filepath_test],
            process_count=2)
        assert train_ds_list[0] == ('one', 1)
        assert train_ds_list[2] == ('three', 3)
        assert len(train_ds_list) == 5

    def test_collect_params(self):
        a_number = 1
        single_argument = 'single_argument'
        list_to_split = ['first', 'second', 'third']
        a_dict = {'test': 'dict'}
        list_not_to_split = [['this', 'lists', 'elements', 'dont', 'split']]
        self.data_interface._set_ranks()
        list_params = self.data_interface._collect_params(
            a_number,
            single_argument,
            list_to_split,
            a_dict,
            list_not_to_split
        )
        assert len(list_params) == 5
        assert list_params[2][0] == list_to_split[0]
        assert list_params[2][1] == list_to_split[1]
        assert list_params[2][2] == list_to_split[2]

        for param in list_params[1]:
            assert param == single_argument

        for param in list_params:
            assert len(param) == 3
