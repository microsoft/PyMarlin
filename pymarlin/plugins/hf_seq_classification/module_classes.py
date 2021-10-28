import os
from pymarlin.plugins.hf_seq_classification.data_classes import HfSeqClassificationDataInterface
import dataclasses

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from transformers import AutoConfig, AutoModelForSequenceClassification
from pymarlin.utils.distributed import rank_zero_only
from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__, "DEBUG")
from pymarlin.plugins import PluginModuleInterface

from .metric_utils import get_metric_func


@dataclasses.dataclass
class ModelArguments:
    encoder_key: str = "bert"
    hf_model: str = "bert-base-uncased"
    model_config_path: str = None
    model_config_file: str = "config.json"
    model_path: str = None
    model_file: str = "pytorch_model.bin"
    tokenizer_path: str = None

@dataclasses.dataclass
class ModuleInterfaceArguments:
    metric: str = "acc"
    max_lr: float = 0.00004  # Maximum learning rate.
    warmup_prop: float = 0.1  # % of steps
    has_labels: bool = True
    max_seq_len: int = 128
    model_args: ModelArguments = None


class HfSeqClassificationModule(PluginModuleInterface):
    """Task specific ModuleInterface used with a trainer.
    The `data` and `model` properties must be set.

    Args:
        ModuleInterface ([type]): [description]
    """

    def __init__(self, args: ModuleInterfaceArguments, data: HfSeqClassificationDataInterface):
        """Initialize training module.

        Args:
            args (arguments.ModuleArguments): Dataclass
            data (HfSeqClassificationDataInterface): PyMarlin data interfsce
        """
        super().__init__()
        self.args = args
        self.data = data
        self.auto_setup(AutoModelForSequenceClassification)
        self.metric_func = get_metric_func(self.args.metric)

    def get_train_dataloader(self, sampler: torch.utils.data.Sampler, batch_size: int):
        train_ds = self.data.get_train_dataset()
        logger.info(f"Training samples = {len(train_ds)}")
        dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=self.collate_fun,
            sampler=sampler(train_ds),
        )
        return dl

    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        val_ds = self.data.get_val_dataset()
        logger.info(f"Validation samples = {len(val_ds)}")
        dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=self.collate_fun,
            sampler=sampler(val_ds),
        )
        return dl

    def collate_fun(self, batch):
        if self.data.args.text_b_col is not None:
            text_a, text_b, label = torch.utils.data._utils.collate.default_collate(batch)
            text_a, text_b, label = list(text_a), list(text_b), list(label)
        else:
            text_a, label = torch.utils.data._utils.collate.default_collate(batch)
            text_a, label = list(text_a), list(label)
            text_b = None
        tokens = self.tokenizer(
            text_a,
            text_b,
            max_length=self.args.max_seq_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )
        tokens['labels'] = torch.tensor(label)
        return tokens

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
