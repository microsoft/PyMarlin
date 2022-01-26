from pymarlin.core.data_interface import DataInterface
from datasets import load_dataset

class SnliData(DataInterface):
    def setup_datasets(self, task):
        self.task = task
        datasets = load_dataset(self.task)
        self.train_ds = datasets['train']
        self.train_ds = self.train_ds.filter(lambda x: x["label"] != -1)
        
        self.val_ds = datasets['validation']
        self.val_ds = self.val_ds.filter(lambda x: x["label"] != -1)

        self.test_ds = datasets['test']
        self.test_ds = self.test_ds.filter(lambda x: x["label"] != -1)
  
    def get_train_dataset(self):
        return self.train_ds

    def get_val_dataset(self):
        return self.val_ds 

    def get_test_dataset(self):
        return self.test_ds