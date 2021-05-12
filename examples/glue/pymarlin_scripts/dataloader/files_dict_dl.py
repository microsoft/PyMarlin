from .base import Base
from torch.utils.data import DataLoader, ConcatDataset
import gc
import os

class FilesDictDataloader(Base):
    def __init__(self, args, datamodule, mode):
        super().__init__(args, mode)
        self.datamodule = datamodule
        self.all_files = [os.path.join(datapath, x) for datapath in self.datapaths for x in sorted(os.listdir(datapath))]

    def get_datacount(self):
        total_samples = 0
        for datafile_path in self.all_files:
            if self.mode == "train":
                dataset = self.datamodule.get_train_dataset(datafile_path)
            elif self.mode == "val":
                dataset = self.datamodule.get_val_dataset(datafile_path)
            total_samples += len(dataset)
            gc.collect()
        return total_samples

    def get_dataloader(self, sampler, batch_size):
        dataloaders_dict = {}
        for datapath in self.datapaths:
            for filename in sorted(os.listdir(datapath)):
                datafile_path = os.path.join(datapath, filename)
                if self.mode == "train":
                    dataset = self.datamodule.get_train_dataset(datafile_path)
                elif self.mode == "val":
                    dataset = self.datamodule.get_val_dataset(datafile_path)
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
                dataloaders_dict[filename] = dataloader
        return dataloaders_dict