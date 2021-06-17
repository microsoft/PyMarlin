import pandas as pd
import torch
import pymarlin
from pymarlin.core import data_interface
import matplotlib
matplotlib.use('Agg') # disable this in local machine to see plots
import matplotlib.pyplot as plt
import sys

def get_source_target(root = 'D:/data/cnn_cln', stage = 'val'):
    source = f'{root}/{stage}.source'
    target = f'{root}/{stage}.target'
    return source, target

class AnalyzeProcessor(data_interface.DataProcessor):
    def __init__(self, source, target):
        with open(source, 'r', encoding = 'UTF-8') as f: 
            self.source = f.readlines()
        with open(target, 'r', encoding = 'UTF-8') as f: 
            self.target = f.readlines()
    def process(self):
        pass
    def analyze(self):
        self.df = pd.DataFrame({'source':self.source, 'target': self.target})
        print(self.df.head())
        print('\nWord length analysis:')
        wordlengths = self.df.applymap(lambda x :  len(x.split()))
        print(wordlengths.describe())
        plt.plot(wordlengths)
        plt.legend(['source','target'])

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        with open(source, 'r', encoding = 'UTF-8') as f: 
            self.source = f.readlines()
        with open(target, 'r', encoding = 'UTF-8') as f: 
            self.target = f.readlines()
        print('len(self.source), len(self.target) = ',len(self.source), len(self.target))
    def __getitem__(self, i):
        # print('len(self.source), len(self.target) = ',len(self.source), len(self.target))
        return self.source[i].strip(), self.target[i].strip()
    def __len__(self):
        return len(self.target)

class SummarizationData(pymarlin.core.data_interface.DataInterface):
    '''
    Class which expects input data to have different files for source and target. 
    Returns dataset which returns non tokenized source and target text.
    '''
    def __init__(self, root= 'D:/data/cnn_cln'):
        self.root = root
        self.train_ds = SummarizationDataset(*get_source_target(root, 'train'))
        self.val_ds = SummarizationDataset(*get_source_target(root, 'val'))
        print('self.train_ds length = ', len(self.train_ds))

    def get_train_dataset(self, *args, **kwargs):
        return self.train_ds
    def get_val_dataset(self, *args, **kwargs):
        return self.val_ds
    def get_test_dataset(self, *args, **kwargs):
        pass

if __name__ == '__main__':
    root = sys.argv[1] #'D:/data/cnn_cln'
    print(root)
    print('\n**** Analyzing Train ***')
    dp = AnalyzeProcessor(*get_source_target(root = root, stage='train'))
    dp.process_data()
    print('\n**** Analyzing Val ***')
    dp = AnalyzeProcessor(*get_source_target(root = root, stage='val'))
    dp.process_data()
    plt.show()