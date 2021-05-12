from .base import Base
from torch.utils.data import DataLoader, ConcatDataset
import gc
import os

'''
    This dataloader loads entire train data per epoch (index). This is different from pretrain dataloaders which don't use entire data per epoch.
    You can control num of data files read into memory at once with args.num_files.
    If num_files=-1 (default), it loads all train files at once.
'''
class FilesGenDataloader(Base):
    def __init__(self, args, datamodule, mode):
        super().__init__(args, mode)
        self.datamodule = datamodule
        self.all_files = [os.path.join(datapath, x) for datapath in self.datapaths for x in sorted(os.listdir(datapath))]
        self.num_files = len(self.all_files) if self.args.num_files == -1 else self.args.num_files

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

    def get_dataloader(self, total_datacount, sampler, batch_size):
        return FilesDL(self.datamodule, self.num_files, self.all_files, total_datacount, sampler, batch_size, self.mode)

class FilesDL(DataLoader):
    def __init__(self, datamodule, num_files, all_files, total_datacount, sampler, batch_size, mode):
        self.file_start_idx = 0
        self.datamodule = datamodule
        self.num_files = num_files
        self.all_files = all_files
        self.mode = mode
        self.sampler = sampler
        self.batch_size = batch_size
        # leftover samples create an extra batch (drop_last=False)
        self.len = (total_datacount // self.batch_size) + 1

    def files_generator(self, all_files, sampler, batch_size):
        while self.file_start_idx < len(all_files):
            datafiles = all_files[self.file_start_idx:self.file_start_idx + self.num_files]
            datasets = []
            for datafile_path in datafiles:
                if self.mode == "train":
                    datasets.append(self.datamodule.get_train_dataset(datafile_path))
                elif self.mode == "val":
                    datasets.append(self.datamodule.get_val_dataset(datafile_path))
            concat_dataset = ConcatDataset(datasets)
            dataloader = DataLoader(concat_dataset, batch_size=batch_size, sampler=sampler(concat_dataset))
            iter_dataloader = iter(dataloader)
            ith_batch = 0
            while ith_batch < len(dataloader):
                batch = next(iter_dataloader)
                yield batch
                ith_batch += 1
            self.file_start_idx += self.num_files

    def __iter__(self):
        return self.files_generator(self.all_files, self.sampler, self.batch_size)

    def __len__(self):
        return self.len