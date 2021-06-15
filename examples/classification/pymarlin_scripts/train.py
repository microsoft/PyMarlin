import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import pymarlin as ml
from pymarlin.utils.logger.logging_utils import getlogger
from pymarlin.utils.stats.basic_stats import StatInitArguments
from pymarlin.utils.writer.base import WriterInitArguments
from pymarlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments

from data import (
    TweetSentData,
    DataInterfaceArguments,
    Stage1,
    Stage2,
)


@dataclass
class ModuleInterfaceArguments:
    max_lr: float = 0.00004
    log_level: str = 'INFO'


class TweetSentModule(ml.ModuleInterface):
    """Tweet sentiment analysis train module
    """

    def __init__(self, data: TweetSentData, args: ModuleInterfaceArguments):
        """

        Args:
            data (TweetSentDataMmodule): data module for this task
            max_lr ([type], optional): maximum learning rate for scheduler. Defaults to 1E-5.
            kwargs (dict): ModuleInterface base class arguments


        Steps:
            1. Call super class constructor
            2. Initialize variables
            3. Define model components
        """
        super().__init__()
        self.args = args
        self.logger = getlogger(__name__, self.args.log_level)
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.reset()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(self.data.get_labels_to_index())
        )

    def reset(self):
        self.lrs = []
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def train_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        outputs = self.model.forward(**batch)
        loss = outputs.loss
        return loss

    def val_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        outputs = self.model.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits, batch.labels

    def get_train_dataloader(self, sampler, batch_size):
        dataset = self.data.get_train_dataset()
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            sampler=sampler(dataset),
        )

    def get_val_dataloaders(self, sampler, batch_size):
        dataset = self.data.get_val_dataset()
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            sampler=sampler(dataset),
        )

    def get_test_dataloaders(self, sampler, batch_size):
        dataset = self.data.get_test_dataset()
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            sampler=sampler(dataset),
        )

    def collate_fn(self, batch):
        tweets, labels = torch.utils.data._utils.collate.default_collate(batch)
        tokens = self.tokenizer(
            list(tweets),
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens["labels"] = labels
        return tokens

    def get_optimizers_schedulers(self, estimated_global_steps_per_epoch: int,
                                  epochs: int):
        optimizer = Adam(self.parameters(), lr=self.args.max_lr)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,
            steps_per_epoch=estimated_global_steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear",
        )
        self.schedulers = scheduler
        return [optimizer], [scheduler]

    def on_end_train_step(self, global_step, train_loss):
        self.lrs.append(self.schedulers.get_last_lr()[0])

    def on_end_train_epoch(self, global_step, train_losses):
        self.train_losses.extend(train_losses)

    def on_end_val_epoch(self, global_step, losses, logits, labels, key="default"):
        """
        args contains all values returned by val_step all_gathered over all processes and all steps
        """
        logits = logits.view(-1, logits.shape[-1])
        preds = torch.argmax(logits, dim = -1).view(-1)

        same = (preds == labels).sum()
        total = labels.shape[0]
        self.val_losses.append(losses.mean())
        self.val_accs.append(same / total)

    def on_end_train(self, global_step):
        self.logger.info("Finished training .. ")
        fig, axes = plt.subplots(2, 2)
        fig.set_figwidth(14)
        fig.set_dpi(150)

        ax = axes[0][0]

        ax.plot(self.train_losses)
        ax.set_xlabel("step")
        ax.set_ylabel("train loss")

        ax = axes[0][1]
        ax.plot(self.val_losses)
        ax.set_xlabel("step")
        ax.set_ylabel("val loss")
        # axes[0].legend = ['train loss','val loss']

        ax = axes[1][0]
        ax.plot(self.val_accs)
        ax.set_xlabel("step")
        ax.set_ylabel("acc")

        ax = axes[1][1]
        ax.plot(self.lrs)
        ax.set_xlabel("step")
        ax.set_ylabel("learning rate")
        plt.show(block=False)


if __name__ == "__main__":

    parser = ml.CustomArgParser(log_level='DEBUG')
    config = parser.parse()

    logger = getlogger(__name__)
    logger.info(f"final merged config = {config}")

    # Instanciate arguments and create DataInterface
    data_args = DataInterfaceArguments(**config['data'])
    data_interface = TweetSentData(data_args)

    # Create DataProcessors
    # stage1 = Stage1(data_args) # assume this is done
    stage2 = Stage2(data_args)

    # Run DataProcessors specifying inputs and ouputs
    # data_interface.process_data(stage1)
    ret = stage2.process_data()

    # Set Datasets and label mappings in DataInterface
    train_ds, val_ds, test_ds, labels_to_index, index_to_labels = ret
    data_interface.setup_datasets(train_ds, val_ds, test_ds,
                              labels_to_index, index_to_labels)

    module_args = ModuleInterfaceArguments(**config['module'])
    module_interface = TweetSentModule(
        data=data_interface,
        args=module_args
    )

    stat_args = StatInitArguments(**config['stat'])
    writer_args = WriterInitArguments(**config['wrts'])
    checkpointer_args = DefaultCheckpointerArguments(**config['chkp'])
    trainer_args = config['trainer']
    trainer_args['stats_args'] = stat_args
    trainer_args['writer_args'] = writer_args
    trainer_args['checkpointer_args'] = checkpointer_args
    trainer_args = ml.TrainerArguments(**trainer_args)

    trainer = ml.Trainer(
        module=module_interface,
        args=trainer_args,
    )

    trainer.train()

    outputs = trainer.validate()
    if trainer.args.distributed_training_args.local_rank == 0:
        print(outputs)
