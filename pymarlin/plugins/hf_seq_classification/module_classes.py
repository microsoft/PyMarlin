import os
import dataclasses

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from transformers import AutoConfig
from pymarlin.utils.distributed import rank_zero_only
from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger.logging_utils import getlogger

logger = getlogger(__name__, "DEBUG")

from pymarlin.core import module_interface, data_interface

from .metric_utils import get_metric_func


@dataclasses.dataclass
class ModelArguments:
    encoder_key: str = "bert"
    hf_model: str = "bert-base-uncased"
    model_config_path: str = None
    model_config_file: str = "config.json"
    model_path: str = None
    model_file: str = "pytorch_model.bin"


@dataclasses.dataclass
class ModuleInterfaceArguments:
    metric: str = "acc"
    max_lr: float = 0.00004  # Maximum learning rate.
    warmup_prop: float = 0.1  # % of steps
    has_labels: bool = True
    model_args: ModelArguments = None


class HfSeqClassificationModule(module_interface.ModuleInterface):
    """Task specific ModuleInterface used with a trainer.
    The `data` and `model` properties must be set.

    Args:
        ModuleInterface ([type]): [description]
    """

    def __init__(self, args):
        """Initialize training module.

        Args:
            args (arguments.ModuleArguments): Dataclass
        """
        super().__init__()
        self.args = args
        self.metric_func = get_metric_func(self.args.metric)

    @property
    def data(self):
        """DataInterface object that is used to retrieve corresponding train or val dataset.

        Returns:
            data (data_interface.DataInterface): DataInterface object with at least one of train or val data.
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
        """Pytorch model.
        Returns:
            model (torch.nn.Module)
        """
        return self._model

    @model.setter
    def model(self, newmodel):
        self._model = newmodel

    def _setup_config(self):
        if self.args.model_args.model_config_path is not None:
            model_config = AutoConfig.from_pretrained(
                os.path.join(
                    self.args.model_args.model_config_path,
                    self.args.model_args.model_config_file,
                )
            )
        else:
            model_config = AutoConfig.from_pretrained(
                self.args.model_args.hf_model
            )
        model_config.num_labels = len(self.data.get_labels())
        return model_config

    def setup_model(self, automodel_class):
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
            student = `HfSeqClassificationModule.model`
            teacher = `HfSeqClassificationModule.teacher`

        Args:
            automodel_class: Huggingface AutoModelForSequenceClassificaton class
        """
        self.model_config = self._setup_config()
        if self.args.model_args.model_path is not None:
            logger.info(f"Model filename: {self.args.model_args.model_file}")
            self.model = automodel_class.from_pretrained(
                os.path.join(
                    self.args.model_args.model_path, self.args.model_args.model_file
                ),
                config=self.model_config,
            )
        else:
            self.model = automodel_class.from_pretrained(
                self.args.model_args.hf_model, config=self.model_config
            )

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
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.args.max_lr,
            steps_per_epoch=estimated_global_steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear",
            pct_start=self.args.warmup_prop,
            div_factor=1e7,  # initial lr ~0
            final_div_factor=1e10,  # final lr ~0
        )
        return [self.optimizer], [self.scheduler]

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
            return outputs.loss, outputs.logits, inputs["labels"]
        else:
            return outputs.logits

    def on_end_train_step(self, global_step, train_loss):
        global_stats.update("lr", self.optimizer.param_groups[0]["lr"], frequent=True)

    def on_end_train(self, global_step):
        logger.info(f"Finished training.")

    @rank_zero_only
    def on_end_val_epoch(self, global_step, *values, key="default"):
        """Compute metrics at the end of each val epoch. Metric function is specified by args.metric.
        values contains all values returned by val_step all_gathered over all processes and all steps
        """
        if self.args.has_labels and len(values) > 0:
            losses, logits, labels = values
            mean_loss = losses.mean().item()
            global_stats.update(key + "/val_loss", mean_loss, frequent=False)
            labels = labels.numpy()
            if len(self.data.get_labels()) == 1:
                preds = logits.squeeze().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).view(-1).numpy()

            logger.info(f"Pred Count = {len(preds)}, Label Count = {len(labels)}")
            assert len(preds) == len(labels)

            metrics = self.metric_func(labels, preds)
            for k in metrics:
                global_stats.update(key + "/val_" + k, metrics[k])
        else:
            logger.info(
                "Either validation data was not provided OR no labels were provided to compute metrics."
            )
