from typing import List, Dict
import torch
from pymarlin.core import trainer_backend, module_interface, trainer
from torch.utils.data import DataLoader

# too long import
from pymarlin.utils.stats import global_stats
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.distributed import rank_zero_only
from pymarlin.utils.logger import getlogger
from pymarlin.plugins import PluginModuleInterface
from transformers import AutoModelForSeq2SeqLM
from torch.optim.lr_scheduler import OneCycleLR
import dataclasses
from .data_classes import HfSeq2SeqData
from .metric_utils import get_metric_func

import re

from filelock import FileLock

try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclasses.dataclass
class ModelArguments:
    encoder_key: str = "bert"
    num_labels: int = None
    hf_model: str = "facebook/bart-base"
    tokenizer_path: str = None
    model_config_path: str = None
    model_config_file: str = "config.json"
    model_path: str = None
    model_file: str = "pytorch_model.bin"
    get_latest_ckpt: bool = True


@dataclasses.dataclass
class GenerateArguments:
    do_sample: bool = False
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False


@dataclasses.dataclass
class ModuleInterfaceArguments:
    max_lr: float = 2e-5  # Maximum learning rate.
    output_mode: str = "s2s"
    max_length_encoder: int = 128
    max_length_decoder: int = 128
    model_args: ModelArguments = ModelArguments()
    generate_args: GenerateArguments = GenerateArguments()
    metric: str = "rouge"


class HfSeq2SeqModule(PluginModuleInterface):
    def __init__(self, data: HfSeq2SeqData, args: ModuleInterfaceArguments):
        super().__init__()
        self.args = args
        self.data = data
        self.auto_setup(AutoModelForSeq2SeqLM)
        self.logger = getlogger(__name__, log_level="DEBUG")
        self.metric_func = get_metric_func(self.args.metric)

    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), self.args.max_lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,
            steps_per_epoch=estimated_global_steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear",
        )
        self.schedulers = scheduler
        return [optimizer], [scheduler]

    def get_train_dataloader(self, sampler: torch.utils.data.Sampler, batch_size: int):
        train_ds = self.data.get_train_dataset()
        dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=self.collate_fun,
            sampler=sampler(train_ds),
        )
        return dl

    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        val_ds = self.data.get_val_dataset()
        dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=self.collate_fun,
            sampler=sampler(val_ds),
        )
        return dl

    def collate_fun(self, batch):
        source, target = torch.utils.data._utils.collate.default_collate(batch)
        # this is probably truncating. repeat positional embeddings and increase embedding layer size.
        source_tokens = self.tokenizer(
            list(source),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_length_encoder,
        )
        target_tokens = self.tokenizer(
            list(target),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_length_decoder,
        )
        labels = target_tokens["input_ids"]
        labels[labels[:, :] == self.model.config.pad_token_id] = -100
        source_tokens["labels"] = labels
        return source_tokens

    def train_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        result = self.model.forward(**batch)
        global_stats.update("lr", self.schedulers.get_last_lr()[0], frequent=True)
        return result.loss

    def val_step(self, global_step: int, batch, device):
        self.logger.debug("inside val_step")
        batch = batch.to(device)
        # result = self.model.forward(**batch)

        self.logger.debug("pre-generate")
        outputs = self.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            max_length=self.args.max_length_decoder,
            **dataclasses.asdict(self.args.generate_args),
        )
        self.logger.debug("post-generate")
        labels = batch.labels
        self.logger.debug("pre-pad")
        labels = self._pad(labels, device)
        outputs = self._pad(outputs, device)
        self.logger.debug("post-pad")
        # print('output size', outputs.size())
        # print('label size', labels.size())
        self.logger.debug("returning...")

        return outputs, labels

    def _pad(self, outputs, device, max_len=None):
        padded_outputs = []
        if max_len is None:
            max_len = self.args.max_length_decoder
        for output in outputs:
            # print('unpadded size', output.size())
            padding = torch.tensor(
                (max_len - len(output)) * [self.tokenizer.pad_token_id]
            ).to(device)
            padded = torch.cat([output, padding])
            # print('padded.size()',padded.size())
            padded_outputs.append(padded)
        return torch.stack(padded_outputs)

    @rank_zero_only
    def on_end_val_epoch(self, global_step, *collated_output, key="default"):
        # move all batch decoding to here so that we can collate tensors in DDP
        self.logger.debug("inside on_end_val_epoch")

        outputs, labels = collated_output
        labels[labels[:, :] == -100] = self.model.config.pad_token_id

        self.logger.debug("pre-decode")
        preds, refs = [], []
        for output, label in zip(outputs, labels):
            try:
                pred = self.tokenizer.decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                ref = self.tokenizer.decode(
                    label, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                preds.append(pred)
                refs.append(ref)
            except Exception as e:
                self.logger.debug("hit decoding exception")
                self.logger.debug(e)

        self.logger.debug("post-decode")
        self.logger.debug(f"len(preds): {len(preds)}")
        self.logger.debug(f"len(refs): {len(refs)}")
        if len(preds) > 0:
            self.logger.debug(f"preds[0]: {preds[0]}")
        if len(refs) > 0:
            self.logger.debug(f"refs[0]: {refs[0]}")
        metrics = self.metric_func(preds, refs)
        for k in metrics:
            global_stats.update(key + "/val_" + k, metrics[k])


if __name__ == "__main__":
    config = CustomArgParser(yaml_file_arg_key="config_path").parse()

    print(config)

    dm = HfSeq2SeqData()
    dm.setup_datasets(root=config["data_path"])

    tm = HfSeq2SeqModule(dm, **config["tm"], generate_kwargs=config["generate"])

    TrainerBackendClass = eval("trainer_backend." + config["trainer_backend_class"])
    tr = TrainerBackendClass()

    tr = trainer_backend.DDPTrainerBackend(tr) if config["dist"] else tr

    tmArgs = trainer.TrainerArguments(
        **config["tmgr"],
        stats_args=trainer.stats.StatInitArguments(**config["stat"]),
        writer_args=trainer.WriterInitArguments(**config["wrt"]),
        checkpointer_args=trainer.DefaultCheckpointerArguments(**config["chkp"]),
    )

    if config["dist"]:
        tr = trainer_backend.DDPTrainerBackend(tr)
    else:
        tmArgs.distributed_training_args = trainer.DistributedTrainingArguments(
            local_rank=config["cuda"]
        )

    trainer = trainer.Trainer(trainer_backend=tr, module=tm, args=tmArgs)
    trainer.validate()
