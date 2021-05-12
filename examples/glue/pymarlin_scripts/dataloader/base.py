import os
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler

class Base:
    STREAMING_COMPAT = False  # mark dataloader as streaming-incompatible by default
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        # self.datapaths below are folder paths for combined and round robin data loader and data file paths for grouped data loader
        self.datasources, self.datapaths = self._get_datasources_and_paths()
        
        # self.logger.info(f"These are the data sources for {self.mode} dataloader: {self.datasources}")
        if mode == 'train': 
            self.check_weights()

        # set up data file tracker for each data source ({datasource1:2} indicates 2 files from datasource 1 have been used for training/evaluating)
        self.datafiles_tracker = {ds:0 for ds in self.datasources}

    def check_weights(self):
        # set up data source weights (default to all 1s if not specified in the argument)
        # current assumption: the weights are given in the same order as the paths (i.e. path1 corresponds to the 1st weight, path2 to the 2nd weight, and so on)
        self.datasource_weights = [1 for d in self.datasources]
        if hasattr(self.args, 'weights') and self.args.weights:
            weights = self.args.weights.strip('[]').split('-')
            if any(['.' in w for w in weights]):
                raise ValueError("Weights should all be integers")
            self.datasource_weights = list(map(int, weights))
        # self.logger.info(f'sampling ratio = {self.datasource_weights},  DataSources = {self.datasources}')

        assert len(self.datasource_weights) == len(self.datasources)

        # filter out data sources with zero weights
        self.datasources = [ds for i, ds in enumerate(self.datasources) if self.datasource_weights[i] != 0]
        self.datapaths = [dp for i, dp in enumerate(self.datapaths) if self.datasource_weights[i] != 0]
        self.datasource_weights = [dw for dw in self.datasource_weights if dw != 0]
        
        self.datasource_weights_ratios = [w/sum(self.datasource_weights) for w in self.datasource_weights]

    def _get_datasources_and_paths(self):
        # This function returns a list of data paths for each data source, for combined and round robin dataloader
        # This method works for combined, combined sampling and round_robin dataloader and will be overriden by grouped dataloader
        datasources = [k for k in vars(self.args).keys() if k.startswith(f'{self.mode}path')]
        datasources = sorted(datasources)
        datapaths = [getattr(self.args, d) for d in datasources]

        # filter out data sources with no specified datapaths
        filtered_datasources, filtered_datapaths = [], []
        for ds, dp in zip(datasources, datapaths):
            if dp:
                filtered_datasources.append(ds)
                filtered_datapaths.append(dp)
                
        return filtered_datasources, filtered_datapaths

    def get_state(self):
        '''
        Fetch dataloader states for each data source, stored in datafiles_tracker
        '''
        return {'dataloader_states': self.datafiles_tracker}

    def restore_state(self, checkpoint_state_dict):
        '''
        Restore dataloader states for each data source
        '''
        if 'dataloader_states' in checkpoint_state_dict:
            self.datafiles_tracker = checkpoint_state_dict['dataloader_states']
            # self.logger.info(f"The dataloader is loaded from last checkpoint with files tracker states {self.datafiles_tracker}")

        
