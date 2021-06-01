import sys
import dataclasses
import os
import pandas as pd
import torch
from pymarlin.core import data_interface
import matplotlib

matplotlib.use("Agg")  # disable this in local machine to see plots
import matplotlib.pyplot as plt

def get_source_target(path="D:/data/cnn_cln", stage="val"):
    source = os.path.join(path, f"{stage}.source")
    target = os.path.join(path, f"{stage}.target")
    return source, target


class AnalyzeProcessor(data_interface.DataProcessor):
    def __init__(self, source, target):
        with open(source, "r", encoding="UTF-8") as f:
            self.source = f.readlines()
        with open(target, "r", encoding="UTF-8") as f:
            self.target = f.readlines()

    def process(self):
        pass

    def analyze(self):
        self.df = pd.DataFrame({"source": self.source, "target": self.target})
        print(self.df.head())
        print("Word length analysis:")
        wordlengths = self.df.applymap(lambda x: len(x.split()))
        print(wordlengths.describe())
        plt.plot(wordlengths)
        plt.legend(["source", "target"])


class HfSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        with open(source, "r", encoding="UTF-8") as f:
            self.source = f.readlines()
        with open(target, "r", encoding="UTF-8") as f:
            self.target = f.readlines()
        print(
            "len(self.source), len(self.target) = ", len(self.source), len(self.target)
        )

    def __getitem__(self, i):
        # print('len(self.source), len(self.target) = ',len(self.source), len(self.target))
        return self.source[i].strip(), self.target[i].strip()

    def __len__(self):
        return len(self.target)


@dataclasses.dataclass
class DataInterfaceArguments:
    data_dir: str = None


class HfSeq2SeqData(data_interface.DataInterface):
    """
    Class which expects input data to have different files for source and target.
    Returns dataset which returns non tokenized source and target text.
    """

    def __init__(self, args: DataInterfaceArguments):
        self.args = args

    def setup_datasets(self):
        self.train_ds = HfSeq2SeqDataset(
            *get_source_target(self.args.data_dir, "train")
        )
        self.val_ds = HfSeq2SeqDataset(*get_source_target(self.args.data_dir, "val"))
        print("self.train_ds length = ", len(self.train_ds))

    def get_train_dataset(self, *args, **kwargs):
        return self.train_ds

    def get_val_dataset(self, *args, **kwargs):
        return self.val_ds


if __name__ == "__main__":
    dm = HfSeq2SeqData()
    root = sys.argv[1]  #'D:/data/cnn_cln'
    print("Train")
    dm.process_data(AnalyzeProcessor(*get_source_target(root=root, stage="train")))
    print("Val")
    dm.process_data(AnalyzeProcessor(*get_source_target(root=root, stage="val")))
    plt.show()

    # dm.setup_datasets()
    # ds = dm.get_train_dataset()
    # len(ds),ds[0]
