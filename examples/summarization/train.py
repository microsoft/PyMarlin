from typing import List,Dict
import torch
from pymarlin.core import trainer_backend, module_interface, trainer
from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch.utils.data import DataLoader

# too long import
from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger import getlogger
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.distributed import rank_zero_only
from torch.optim.lr_scheduler import OneCycleLR

from data import SummarizationData
from rouge_score import rouge_scorer,scoring
import re

from filelock import FileLock

logger = getlogger(__file__)

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
            drop_last=True, # ORT fix, batch size needs to stay constant
        )
        return dl

    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        val_ds = self.data.get_val_dataset()
        dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=self.collate_fun,
            sampler=sampler(val_ds),
            drop_last=True, # ORT fix, batch size needs to stay constant
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
            padding='max_length', # To make all tensor seq length dimension of same length for collation
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length_decoder,
        )
        labels = target_tokens["input_ids"]
        labels[labels[:, :] == self.pad_token_id] = -100
        source_tokens["labels"] = labels
        return source_tokens
    
    @property
    def pad_token_id(self):
        return self.model.config.pad_token_id

    def train_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        result = self.model.forward(**batch)
        global_stats.update("lr", self.schedulers.get_last_lr()[0], frequent=True)
        return result.loss

    def val_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        # result = self.model.forward(**batch)
        summaries = self.model.generate(
           input_ids=batch.input_ids, 
            attention_mask=batch.attention_mask,
            **self.generate_kwargs
        )
        labels = batch.labels
        labels[labels[:, :] == -100] = self.pad_token_id
        # pad summaries till same length for gathering
        # Idle solution will be calculate ROUGE in a distributed manner if possible. Will save gather cost
        padded_summaries = torch.ones_like(labels) * self.pad_token_id
        padded_summaries[:,:summaries.shape[-1]] = summaries
        return padded_summaries, labels

    def calculate_rouge(
        self,
        pred_lns: List[str],
        tgt_lns: List[str],
        use_stemmer=True,
        rouge_keys=["rouge1", "rouge2", "rougeL"],
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=True,
    ) -> Dict:
        """Calculate rouge using rouge_scorer package.
        Args:
            pred_lns: list of summaries generated by model
            tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
            use_stemmer:  Bool indicating whether Porter stemmer should be used to
            strip word suffixes to improve matching.
            rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
            return_precision_and_recall: (False) whether to also return precision and recall.
            bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
                this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
            newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
            on multi sentence summaries (CNN/DM dataset).
        Returns:
            Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys
        """
        def extract_rouge_mid_statistics(dct):
            new_dict = {}
            for k1, v1 in dct.items():
                mid = v1.mid
                new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
            return new_dict

        def add_newline_to_end_of_each_sentence(x: str) -> str:
            """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
            re.sub("<n>", "", x)  # remove pegasus newline char
            assert NLTK_AVAILABLE, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
            return "\n".join(nltk.sent_tokenize(x))

        scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
        aggregator = scoring.BootstrapAggregator()
        for pred, tgt in zip(tgt_lns, pred_lns):
            # rougeLsum expects "\n" separated sentences within a summary
            if newline_sep:
                pred = add_newline_to_end_of_each_sentence(pred)
                tgt = add_newline_to_end_of_each_sentence(tgt)
            scores = scorer.score(pred, tgt)
            aggregator.add_scores(scores)

        if bootstrap_aggregation:
            result = aggregator.aggregate()
            if return_precision_and_recall:
                return extract_rouge_mid_statistics(result)  # here we return dict
            else:
                return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
        else:
            return aggregator._scores  # here we return defaultdict(list)

    @rank_zero_only
    def on_end_val_epoch(self, global_step, *collated_output, key="default"):
        logger.log('Evaluating gathered results.')
        if len(collated_output) == 0:
            logger.error("len(collated_output) == 0)")
            return 
        summaries, labels = collated_output
        #decode
        preds = self.tokenizer.batch_decode(
            summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        refs = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.log(f"preds[:2]: {preds[:2]}")
        logger.log(f"refs[:2]: {refs[:2]}")
        ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scores: dict =  self.calculate_rouge(preds, refs, rouge_keys = ROUGE_KEYS)
        global_stats.update_multi('metrics/rouge', scores)
        print(scores)

if __name__ == '__main__':
    config = CustomArgParser(yaml_file_arg_key="config_path").parse()

    print(config)

    # data interface
    dm = SummarizationData()
    dm.setup_datasets(root=config["data_path"])

    # Training Module Interface
    summarization_module = SummarizationBartModule(dm, **config["module"], generate_kwargs=config["generate"])

    trainer_args = trainer.TrainerArguments(
        **config["trainer"],
        stats_args=trainer.stats.StatInitArguments(**config["stat"]),
        writer_args=trainer.WriterInitArguments(**config["wrt"]),
        checkpointer_args=trainer.DefaultCheckpointerArguments(**config["chkp"])
    )

    trainer = trainer.Trainer(module=summarization_module, args=trainer_args)

    trainer.validate()
    trainer.train()
