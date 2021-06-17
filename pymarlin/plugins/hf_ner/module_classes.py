import os
import dataclasses
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from pymarlin.core import module_interface, data_interface
from transformers import AutoModelForTokenClassification

from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger.logging_utils import getlogger
from .sequence_labelling_metrics import get_ner_seq_metric
from pymarlin.utils.distributed import rank_zero_only
from pymarlin.plugins.hf_ner.data_classes import NERDataInterface
from pymarlin.plugins import PluginModuleInterface

logger = getlogger(__name__, "DEBUG")


@dataclasses.dataclass
class ModelArguments:
    model_name: "bert"
    encoder_key: "bert"
    hf_model: "bert-base-uncased"
    model_file: "pytorch_model.bin"
    model_config_file: "config.json"
    model_path: None
    model_config_path: None
    tokenizer_path: None

@dataclasses.dataclass
class ModuleInterfaceArguments:
    output_dir: None
    max_lr: 0.00004  # Maximum learning rate.
    warmup_prop: 0.1  # % of steps
    has_labels: True
    max_seq_len: 128
    pad_label_id: -100
    label_all_tokens: False
    model_args: ModelArguments = ModelArguments

class NERModule(PluginModuleInterface):
    """NER Task specific ModuleInterface used with a trainer.
    The `data` and `model` are required properties and must be set.

    Args:
        ModuleInterfaceArguments : contains module interface arguments , i.e. max learning rate,
        warmup propotion, type of trainer , etc. Also includes modelArguments class as attribute
        which include model specific arguments such as hfmodel name , modep path , model file name , etc
    """

    def __init__(self, args, data: NERDataInterface):
        super().__init__()
        self.args = args
        self.metric_func = get_ner_seq_metric
        self.data = data
        self.auto_setup(AutoModelForTokenClassification)
   
    
    def get_train_dataloader(self, sampler: torch.utils.data.Sampler, batch_size: int):
        train_ds = self.data.get_train_dataset()
        logger.info(f"Training samples = {len(train_ds)}")
        dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=self.collate_func,
            sampler=sampler(train_ds),
        )
        return dl

    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        val_ds = self.data.get_val_dataset()
        logger.info(f"Validation samples = {len(val_ds)}")
        dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=self.collate_func,
            sampler=sampler(val_ds),
        )
        return dl

    def collate_func(self,batch):
        sentence, labels = zip(*batch)
        sentence, labels = list(sentence), list(labels)

        tokenized_inputs = self.tokenizer(
            sentence,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True,
            max_length=self.args.max_seq_len,
        )

        label_ids = []
        for i in range(len(sentence)): # for each sentence in input            
            if self.args.has_labels:
                current_label_ids = []
                current_label = labels[i]
                word_ids = tokenized_inputs.word_ids(i)
                prev_word_idx = None  # To track subwords
                for word_idx in word_ids:
                    if word_idx is None:  # special tokens have None
                        current_label_ids.append(self.args.pad_label_id)
                    elif (word_idx != prev_word_idx):  # First part of a word always gets the label
                        current_label_ids.append(self.data.label_map[current_label[word_idx]])
                    else:  # other subword tokens get the same label or ignore index, controlled by flag label_all_tokens
                        current_label_ids.append(
                            self.data.label_map[current_label[word_idx]]
                            if self.args.label_all_tokens
                            else self.args.pad_label_id
                        )
                    prev_word_idx = word_idx
                label_ids.append(current_label_ids)

        tokenized_inputs['labels'] = torch.tensor(label_ids)
        return tokenized_inputs

    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
    ):
        self.optimizer = Adam(self.model.parameters(), self.args.max_lr)
        self.schedulers = OneCycleLR(
            self.optimizer,
            max_lr=self.args.max_lr,
            steps_per_epoch=estimated_global_steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear",
            pct_start=self.args.warmup_prop,
            div_factor=1e7,  # initial lr ~0
            final_div_factor=1e10,  # final lr ~0
        )
        return [self.optimizer], [self.schedulers]

    def _inputs_to_device(self, batch, device):
        inputs = {}
        for k, v in batch.items():
            if v is not None:
                inputs[k] = v.to(device)
        return inputs

    def train_step(self, global_step, batch, device):
        batch = self._inputs_to_device(batch, device)
        outputs = self.model.forward(**batch)
        loss = outputs.loss
        return loss

    def val_step(self, global_step, batch, device):
        batch = self._inputs_to_device(batch, device)
        outputs = self.model.forward(**batch)
        if self.args.has_labels:
            return outputs.loss, outputs.logits, batch["labels"]
        else:
            return outputs.logits

    def on_end_train_step(self, global_step, train_loss):
        global_stats.update("lr", self.optimizer.param_groups[0]["lr"], frequent=True)

    @rank_zero_only
    def on_end_val_epoch(self, global_step, *inputs, key="default"):
        if self.args.has_labels and len(inputs) > 0:
            loss, logits, labels = inputs

            logits = logits.cpu().numpy()
            logits = logits.reshape(-1, logits.shape[-1])
            predictions = np.argmax(logits, axis=1)

            label_ids = labels.to("cpu").numpy().reshape(-1)

            str_preds = [
                self.data.get_labels()[int(p)]
                for (p, l) in zip(predictions, label_ids)
                if l != self.args.pad_label_id
            ]
            str_labels = [
                self.data.get_labels()[int(l)]
                for (p, l) in zip(predictions, label_ids)
                if l != self.args.pad_label_id
            ]

            metrics = self.metric_func(str_labels, str_preds)
            for k in metrics:
                global_stats.update(k, metrics[k])
        else:
            logger.info(
                "Either validation data was not provided OR no labels were provided to compute metrics."
            )
