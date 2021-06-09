import os
import dataclasses
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from pymarlin.core import module_interface, data_interface
from transformers import AutoConfig

from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger.logging_utils import getlogger
from .sequence_labelling_metrics import get_ner_seq_metric
from pymarlin.utils.distributed import rank_zero_only

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


@dataclasses.dataclass
class ModuleInterfaceArguments:
    tr_backend: "singleprocess"
    output_dir: None
    max_lr: 0.00004  # Maximum learning rate.
    warmup_prop: 0.1  # % of steps
    has_labels: True
    model_args: ModelArguments = ModelArguments


class NERModule(module_interface.ModuleInterface):
    """NER Task specific ModuleInterface used with a trainer.
    The `data` and `model` are required properties and must be set.

    Args:
        ModuleInterfaceArguments : contains module interface arguments , i.e. max learning rate,
        warmup propotion, type of trainer , etc. Also includes modelArguments class as attribute
        which include model specific arguments such as hfmodel name , modep path , model file name , etc
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.metric_func = get_ner_seq_metric

    @property
    def data(self):
        """DataInterface object that is used to retrieve corresponding train or val dataset.

        Returns:
            data: DataInterface object with at least one of train or val data.
        """
        assert (
            len(self._data.get_train_dataset()) != 0
            or len(self._data.get_val_dataset()) != 0
        )
        return self._data

    @data.setter
    def data(self, datainterface):
        assert isinstance(datainterface, data_interface.DataInterface)
        assert (
            len(datainterface.get_train_dataset()) != 0
            or len(datainterface.get_val_dataset()) != 0
        )
        self._data = datainterface

    @property
    def model(self):
        """Pytorch model."""
        return self._model

    @model.setter
    def model(self, newmodel):
        self._model = newmodel

    def _setup_config(self):
        if self.args.model_args.model_config_path is not None:
            self.model_config = AutoConfig.from_pretrained(
                os.path.join(
                    self.args.model_args.model_config_path,
                    self.args.model_args.model_config_file,
                )
            )
        else:
            self.model_config = AutoConfig.from_pretrained(
                self.args.model_args.hf_model
            )

        self.model_config.num_labels = len(self.data.get_labels())

    def setup_model(self, model_class):
        """Initializes `HfSeqClassificationModule.model` weights:
            Option 1: Load weights from specified files mentioned in YAML config
                        model:
                            model_config_path
                            model_config_file
                            model_path
                            model_file
            Option 2: Load from Huggingface model hub, specify string in YAML config as:
                        model:
                            hf_model
        If distill_args.enable = True
            student = `NERModule.model`
            teacher = `NERModule.teacher`

        Args:
            automodel_class: Huggingface AutoModelForTokenClassificaton class
        """
        self._setup_config()
        if self.args.model_args.model_path is not None:
            self.model = model_class.from_pretrained(
                os.path.join(
                    self.args.model_args.model_path, self.args.model_args.model_file
                ),
                config=self.model_config,
            )
        else:
            self.model = model_class.from_pretrained(
                self.args.model_args.hf_model, config=self.model_config
            )
        return self.model

    def get_train_dataloader(self, sampler: torch.utils.data.Sampler, batch_size: int):
        train_ds = self.data.get_train_dataset()
        logger.info(f"Training samples = {len(train_ds)}")
        dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler(train_ds),
        )
        return dl

    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        val_ds = self.data.get_val_dataset()
        logger.info(f"Validation samples = {len(val_ds)}")
        dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=sampler(val_ds),
        )
        return dl

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
        logger.debug(f"Loss = {loss.item()}")
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
                if l != self.data.args.pad_label_id
            ]
            str_labels = [
                self.data.get_labels()[int(l)]
                for (p, l) in zip(predictions, label_ids)
                if l != self.data.args.pad_label_id
            ]

            metrics = self.metric_func(str_labels, str_preds)
            for k in metrics:
                global_stats.update(k, metrics[k])
        else:
            logger.info(
                "Either validation data was not provided OR no labels were provided to compute metrics."
            )
