from typing import List,Dict
import torch
from pymarlin.core import trainer_backend, module_interface, trainer
from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch.utils.data import DataLoader

# too long import
from pymarlin.utils.stats import global_stats
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from torch.optim.lr_scheduler import OneCycleLR

from data import SummarizationData
from rouge_score import rouge_scorer,scoring
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

class SummarizationBartModule(module_interface.ModuleInterface):
    def __init__(
        self,
        data: SummarizationData,
        max_length_encoder=128,
        max_length_decoder=128,
        max_lr=2e-5,
        generate_kwargs = {}
    ):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.max_lr = max_lr
        self.max_length_encoder = max_length_encoder
        self.max_length_decoder = max_length_decoder
        self.generate_kwargs = generate_kwargs
        self.data = data

    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), self.max_lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
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
            max_length=self.max_length_encoder,
        )
        target_tokens = self.tokenizer(
            list(target),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length_decoder,
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
        batch = batch.to(device)
        # result = self.model.forward(**batch)
        summaries = self.model.generate(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        )
        preds = self.tokenizer.batch_decode(
            summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        labels = batch.labels
        labels[labels[:, :] == -100] = self.model.config.pad_token_id
        refs = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return preds, refs

    def on_end_val_epoch(self, global_step, *collated_output, key="default"):
        preds, refs = collated_output
        print(refs)
        print(preds)

config = CustomArgParser(yaml_file_arg_key="config_path").parse()

print(config)

dm = SummarizationData()
dm.setup_datasets(root=config["data_path"])

tm = SummarizationBartModule(dm, **config["tm"], generate_kwargs=config["generate"])

TrainerBackendClass = eval("trainer_backend." + config["trainer_backend_class"])
tr = TrainerBackendClass()

tr = trainer_backend.DDPTrainerBackend(tr) if config["dist"] else tr

tmArgs = trainer.TrainerArguments(
    **config["tmgr"],
    stats_args=trainer.stats.StatInitArguments(**config["stat"]),
    writer_args=trainer.WriterInitArguments(**config["wrt"]),
    checkpointer_args=trainer.DefaultCheckpointerArguments(**config["chkp"])
)

if config["dist"]:
    tr = trainer_backend.DDPTrainerBackend(tr)
else:
    tmArgs.distributed_training_args = trainer.DistributedTrainingArguments(
        local_rank = config["cuda"]
        )

trainer = trainer.Trainer(trainer_backend=tr, module=tm, args=tmArgs)


trainer.validate()
