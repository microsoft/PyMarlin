import os
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import Dataset
from pymarlin.core.data_interface import DataInterface, DataProcessor
from pymarlin.utils.logger.logging_utils import getlogger


@dataclass
class DataInterfaceArguments:
    filepath_train: str = '../data/Corona_NLP_train.csv'
    filepath_test: str = '../data/Corona_NLP_test.csv'
    preprocessed_dir: str = '../preprocessed'
    encoding: str = 'ISO-8859-1'
    text_field: str = 'OriginalTweet'
    label_field: str = 'Sentiment'
    splitpct: int = 10
    log_level: str = "INFO"


class TweetDataset(Dataset):
    def __init__(self, df, labels_to_index,
                 text_field, label_field):
        self.df = df
        self.labels = labels_to_index
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (row[self.text_field], self.labels[row[self.label_field]])


class Stage1(DataProcessor):

    def __init__(self, args):
        self.args = args
        self.logger = getlogger(__name__, log_level=self.args.log_level)

    # This function doesnt need any inputs other than those from DataInterfaceArguments
    # Doesn't return anything and instead saves results to file
    def process(self, filename=None):
        if filename:
            self._read_and_save(filename)
        else:
            self._read_and_save(self.args.filepath_train)
            self._read_and_save(self.args.filepath_test)
        self.logger.info('Stage 1 preprocessing successful')

    def _read_and_save(self, filepath):
        df = pd.read_csv(filepath,
                         encoding=self.args.encoding)

        #Remove unnecessary columns (dummy preprocessing)
        df = df[[self.args.text_field, self.args.label_field]]
        print(f'Size of {filepath} = {len(df)}')
        self.logger.debug(df.head())

        # Save
        filename = filepath.split('\\')[-1].split('.')[0]
        os.makedirs(self.args.preprocessed_dir, exist_ok=True)
        df.to_csv(os.path.join(self.args.preprocessed_dir, filename+'_stage1.csv'))


class Stage2(DataProcessor):

    def __init__(self, args):
        self.args = args
        self.logger = getlogger(__name__, log_level=self.args.log_level)
        self.train_df = None

    def create_dataset(self, df, labels_to_index,
                       text_field, label_field):
        return TweetDataset(df, labels_to_index,
                            text_field, label_field)

    # This function doesnt need any inputs other than those from DataInterfaceArguments
    def process(self):
        train_filename = self.args.filepath_train.split('\\')[-1].split('.')[0]
        test_filename = self.args.filepath_test.split('\\')[-1].split('.')[0]
        self.train = pd.read_csv(os.path.join(self.args.preprocessed_dir,
                                              train_filename+'_stage1.csv'),
                                 encoding=self.args.encoding)
        test = pd.read_csv(os.path.join(self.args.preprocessed_dir,
                                        test_filename+'_stage1.csv'),
                           encoding=self.args.encoding)

        unique_labels = pd.concat([self.train,test])[self.args.label_field].unique()
        labels_to_index = {label: i for i, label
                           in enumerate(unique_labels)}
        index_to_labels = {i:label for i, label in labels_to_index.items()}
        self.logger.debug(labels_to_index)

        # Train test split and dataset creation
        train_ds = self.create_dataset(self.train, labels_to_index,
                                       self.args.text_field, self.args.label_field)
        val_ds_size = len(train_ds)*self.args.splitpct//100
        train_ds, val_ds = random_split(train_ds,[len(train_ds)-val_ds_size,val_ds_size])

        test_ds = self.create_dataset(test, labels_to_index,
                                      self.args.text_field, self.args.label_field)

        self.logger.debug('train_ds: %i, val_ds: %i, test_ds: %i' %
                          (len(train_ds), len(val_ds), len(test_ds)))
        self.logger.info('Stage 2 preprocessing successful\n')

        return train_ds, val_ds, test_ds, labels_to_index, index_to_labels

    def analyze(self, *args):
        # pd.set_option('display.max_colwidth', 50)
        self.logger.info(self.train.head())

        plt.style.use('ggplot')
        _ = plt.hist([len(t.split(' ')) for t in self.train[self.args.text_field]])
        plt.title('words in tweet')
        plt.show(block = False)

        plt.figure()
        _ = plt.hist(self.train[self.args.label_field])
        plt.title('Sentiments')
        plt.show(block = False)


class TweetSentData(DataInterface):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = getlogger(__name__, log_level=self.args.log_level)
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self._labels_to_index = None
        self._index_to_labels = None

    def setup_datasets(self, train_ds, val_ds, test_ds,
                       labels_to_index, index_to_labels):
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
        self._labels_to_index = labels_to_index
        self._index_to_labels = index_to_labels

    def get_train_dataset(self):
        return self._train_ds

    def get_val_dataset(self):
        return self._val_ds

    def get_test_dataset(self):
        return self._test_ds

    def get_labels_to_index(self):
        return self._labels_to_index

    def get_index_to_labels(self):
        return self._index_to_labels


if __name__ == '__main__':

    ## ----------------------- SINGLE PROCESS ------------------- ##

    # Instanciate arguments and create DataInterface
    data_args = DataInterfaceArguments()
    data_interface = TweetSentData(data_args)

    # Create DataProcessors
    stage1 = Stage1(data_args)
    stage2 = Stage2(data_args)

    # Run DataProcessors specifying inputs and ouputs
    stage1.process_data()
    ret = stage2.process_data()

    # Set Datasets and label mappings in DataInterface
    train_ds, val_ds, test_ds, labels_to_index, index_to_labels = ret
    data_interface.setup_datasets(train_ds, val_ds, test_ds,
                              labels_to_index, index_to_labels)

    plt.show()

    ## ----------------------- MULTI PROCESS ------------------- ##

    data_args = DataInterfaceArguments()
    data_interface = TweetSentData(data_args)
    stage1 = Stage1(data_args)
    stage2 = Stage2(data_args)
    stage1.multi_process_data(
        [data_args.filepath_train, data_args.filepath_test],
        process_count=2,
        )
    ret = stage2.process_data()
    train_ds, val_ds, test_ds, labels_to_index, index_to_labels = ret
    data_interface.setup_datasets(
        train_ds, val_ds, test_ds,
        labels_to_index, index_to_labels,)
    plt.show()
