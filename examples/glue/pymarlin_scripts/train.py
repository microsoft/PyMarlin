import os, sys
import dataclasses
import json
import argparse
import pickle
import csv
import gc

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

import transformers
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModelForSequenceClassification

from pymarlin.core import trainer, trainer_backend, module_interface
from pymarlin.utils.stats import global_stats
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.logger.logging_utils import getlogger

from dataloader.files_gen_dl import FilesGenDataloader
from dataloader.files_dict_dl import FilesDictDataloader
from data import TaskData, DataInterfaceArguments, InputFeatures
from glue_processors import glue_dict
from init_args import ModuleInterfaceArguments, ModelArgs

class Recipe(module_interface.ModuleInterface):
    def __init__(self, args, datamodule, logger):
        super().__init__()
        self.args = args
        self.datamodule = datamodule
        self.logger = logger

    def _setup_configs(self):
        self.model_config = AutoConfig.from_pretrained(
            os.path.join(self.args.model_args.model_config_path, self.args.model_args.model_config_file)
        )
        if self.args.model_args.num_labels is not None:
            self.model_config.num_labels = self.args.model_args.num_labels

    def setup_models(self):
        if self.args.model_args.model_wts_path is not None:
            self._setup_configs()
            if self.args.model_args.get_latest_ckpt:
                filenames = [f for f in os.listdir(self.args.model_args.model_wts_path) if (f.endswith('.pt') and not f.endswith('ort.pt')) or f.endswith('.bin')]
                self.args.model_args.model_file = max(filenames)
                self.logger.info(f"Model filename from get_latest_ckpt: {self.args.model_args.model_file}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(self.args.model_args.model_wts_path, self.args.model_args.model_file),
                config=self.model_config
                )
        else:
            self.model_config = AutoConfig.from_pretrained(self.args.model_args.hf_model)
            if self.args.task.lower() == 'sts-b': # STS-B is a regression task
                self.model_config.num_labels = 1
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_args.hf_model, config=self.model_config)
        return self.model
        
    def get_train_dataloader(self, sampler, batch_size):
        dl = FilesGenDataloader(self.args, self.datamodule, "train")
        total_datacount = dl.get_datacount()
        self.logger.info(f"Total training samples = {total_datacount}")
        dl_gen = dl.get_dataloader(total_datacount, sampler, batch_size)
        return dl_gen

    def get_val_dataloaders(self, sampler, batch_size):
        dl = FilesDictDataloader(self.args, self.datamodule, "val")
        total_datacount = dl.get_datacount()
        self.logger.info(f"Total validation samples = {total_datacount}")
        dl_dict = dl.get_dataloader(sampler, batch_size)
        return dl_dict

    def get_optimizers_schedulers(self, estimated_global_steps_per_epoch: int, epochs: int):
        optimizer = Adam(self.model.parameters(), self.args.max_lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,
            steps_per_epoch=estimated_global_steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear",
            pct_start=self.args.warmup_prop,
            div_factor=1e7,# initial lr ~0
            final_div_factor=1e10 # final lr ~0
        )
        self.schedulers = scheduler
        self.optimizers = optimizer
        return [optimizer], [scheduler]
        
    def _inputs_to_device(self, batch, device):
        inputs = {}
        for k, v in batch.items():
            if v is not None:
                inputs[k] = v.to(device)
        return inputs

    def train_step(self, global_step, batch, device):
        inputs = self._inputs_to_device(batch, device)
        outputs = self.model.forward(**inputs)
        loss = outputs.loss
        return loss
        
    def val_step(self, global_step, batch, device):
        inputs = self._inputs_to_device(batch, device)
        outputs = self.model.forward(**inputs)
        if outputs.loss is not None:
            return outputs.loss, outputs.logits, inputs['labels']
        else:
            return outputs.logits
    
    def on_end_train_step(self, global_step, train_loss):
        # global_stats.update("lr", self.schedulers.get_last_lr()[0], frequent=True)
        global_stats.update("lr", self.optimizers.param_groups[0]["lr"], frequent=True)

    def on_end_train(self, global_step):
        self.logger.info(f"Finished training.")
        # self.logger.info(f"Saving model and config to {self.args.output_dir}.")
        # torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, self.args.model_args.model_file))
        # self.model_config.to_json_file(os.path.join(self.args.output_dir, self.args.model_args.model_config_file), use_diff=False)

    def on_end_val_epoch(self, global_step, *inputs, key="default"):
        """
        args contains all values returned by val_step all_gathered over all processes and all steps
        """
        if not self.args.no_labels:
            losses, logits, labels = inputs
            mean_loss = losses.mean().item()
            global_stats.update(key+'/val_loss', mean_loss, frequent=False)
            labels = labels.numpy()
            if self.args.task.lower() == 'sts-b':
                preds = logits.squeeze().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).view(-1).numpy()

            self.logger.info(f"Pred Count = {len(preds)}, Label Count = {len(labels)}")
            assert len(preds) == len(labels)

            metrics = glue_dict[self.args.task.lower()]["metric"](labels, preds)
            for k in metrics:
                global_stats.update(key+'/val_'+k, metrics[k])


if __name__ == "__main__":

    parser = CustomArgParser(log_level='DEBUG')
    config = parser.parse()
    logger = getlogger(__name__, config['module']['log_level'])

    logger.info(f"final merged config = {config}\n")

    os.makedirs(config['module']['output_dir'], exist_ok=True)

    data = TaskData(DataInterfaceArguments(**config['dmod']))
    model_args = ModelArgs(**config['model'])
    config['module']['model_args'] = model_args
    recipe_args = ModuleInterfaceArguments(**config['module'])
    recipe = Recipe(recipe_args, data, logger)
    model = recipe.setup_models()

    trainer_args = trainer.TrainerArguments(
        **config["trainer"],
        stats_args=trainer.stats.StatInitArguments(**config['stats']),
        writer_args=trainer.WriterInitArguments(**config['wrts']),
        checkpointer_args=trainer.DefaultCheckpointerArguments(**config['ckpt'])
    )
    
    if recipe_args.fp16:
        backend = trainer_backend.SingleProcessAmp()
    else:
        backend = trainer_backend.SingleProcess()
    logger.debug(f"Trainer backend = {backend}")

    if recipe_args.trainer_backend not in ['SingleProcess', 'SingleProcessAmp']:
        backend = getattr(trainer_backend, recipe_args.trainer_backend)(backend)
    glue_trainer = trainer.Trainer(trainer_backend=backend, module=recipe, args=trainer_args)
    getattr(glue_trainer, recipe_args.operation)()

# train
# python train.py --tmod.operation "train" --tmod.task "RTE" --tmod.trainpath "processed_data\RTE\train" --tmod.valpath "processed_data\RTE\dev" --tmod.output_dir "train_out" --model.hf_model "bert-base-uncased"
# inference
# python train.py --tmod.operation "validate" --tmod.task "RTE" --tmod.valpath "processed_data\RTE\dev" --tmod.output_dir "dev_out" --model.model_wts_path "train_out" --model.model_config_path "train_out"

# distillation inference
# python train.py --tmod.task "RTE" --tmod.valpath "processed_data\RTE\dev" --tmod.output_dir "distill_dev_out" --model.model_wts_path "distill_out" --model.model_config_path "distill_out"

