from pymarlin.core.module_interface import ModuleInterface
from pymarlin.core.trainer import Trainer, TrainerArguments, WriterInitArguments,DistributedTrainingArguments,DefaultCheckpointerArguments
from pymarlin.utils.stats.basic_stats import StatInitArguments
from pymarlin.core.trainer_backend import SingleProcess,SingleProcessAmp, DDPTrainerBackend, SingleProcessApexAmp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.stats import global_stats
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    accuracy_score
)

from scipy.stats import spearmanr,pearsonr
from typing import Union
class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.config = encoder.config
        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels) # truncated normal with initializer range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, encoder_output
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        hidden_state = encoder_output[0]  # (bs, seq_len, dim)
        # pooled_output = hidden_state[:, 0]  # (bs, dim) # CLS token only
        pooled_output = encoder_output[1] # (bs, dim) This is after going through the pooler layer in encoder
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
            # hidden_states=encoder_output.hidden_states, # not supported in turing models
            # attentions=encoder_output.attentions,
        )


class SentenceClassifier(ModuleInterface):
    def __init__(self, 
    data_interface, 
    max_lr = 2e-5, 
    num_labels = 2, 
    encoder:Union[nn.Module, str] = "bert-base-uncased", 
    tokenizer = "bert-base-uncased", 
    max_length = 512,
    warmup = 0.1,
    head_only = False):
        '''
            Calssifier for single sentence.
            Assumes data_interface dataset to return label and sentence.

        '''
        super().__init__()  # always initialize superclass first
        self.data_interface = data_interface
        self.max_lr = max_lr
        self.max_length = max_length
        self.num_labels = num_labels
        self.warmup = warmup
        self.glue_task = self.data_interface.glue_task
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer 

        self.criterion = nn.CrossEntropyLoss()

        if type(encoder) == str:
           encoder = AutoModel.from_pretrained(encoder) # this will be optimized too if str is passed.
           head_only = False
        
        if not head_only:
            self.encoder = encoder # making self.encoder will optimize the head too
        
        self.net = Classifier(encoder, self.num_labels)
        
        self.optimizer = optim.Adam(self.parameters(), lr=max_lr)


    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
    ):
        # print('\n\n\nestimated_global_steps_per_epoch',estimated_global_steps_per_epoch)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=estimated_global_steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear",
            pct_start = self.warmup,
            div_factor=1e7,# initial lr ~0
            final_div_factor=1e10 # final lr ~0
        )
        return [self.optimizer], [self.scheduler]

    def get_train_dataloader(self, sampler: type, batch_size: int):
        ds = self.data_interface.get_train_dataset()
        return torch.utils.data.DataLoader(
            ds,
            sampler = sampler(ds),
            batch_size=batch_size,
            collate_fn=self.collate_fun,
        )

    def collate_fun(self, batch):
        # print(batch)
        batch = torch.utils.data._utils.collate.default_collate(batch)
        # print(batch)
        labels = batch["label"]
        sentences = batch["sentence"]
        input = self.tokenizer(
            sentences,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return input, torch.LongTensor(labels)

    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        ds = self.data_interface.get_val_dataset()
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=self.collate_fun,
            sampler =sampler(ds)
        )

    def train_step(self, global_step: int, batch, device="cpu", encoder = None):
        """
        First output should be loss. Can return multiple outputs
        """
        global_stats.update(f"lr/{self.glue_task}", self.scheduler.get_last_lr()[0], frequent=True)

        encoder = encoder if encoder else self.encoder
        inputs, labels = batch  # output of dataloader will be input of train_step
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = self.net(encoder(**inputs))

        # print(outputs)
        if self.num_labels ==1:
            outputs.logits = outputs.logits.squeeze()
        loss = self.criterion(outputs.logits, labels)
        global_stats.update(f"loss/{self.glue_task}", loss.item(), frequent = True)
        return loss

    def val_step(self, global_step: int, batch, device="cpu", encoder = None):
        """
        Can return multiple outputs. First output need not be loss.
        """
        # print(batch)
        encoder = encoder if encoder else self.encoder
        inputs, labels = batch  # output of dataloader will be input of train_step
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = self.net(encoder(**inputs))
        if self.num_labels ==1:
            outputs.logits = outputs.logits.squeeze()
        loss = self.criterion(outputs.logits, labels)
        if self.num_labels ==1:
            predicted = outputs.logits.data
        else:
            _, predicted = torch.max(outputs.logits.data, 1)
        return loss, predicted, labels

    def on_end_val_epoch(
        self, global_step: int, *val_step_collated_outputs, key="default"
    ):
        """
        callback after validation loop ends
        """
        loss, predicted, labels = val_step_collated_outputs
        # print(labels, predicted)
        loss = loss.mean().item()
        acc = accuracy_score(labels, predicted)
        mcc = matthews_corrcoef(labels, predicted)
        f1 = f1_score(labels,predicted)
        global_stats.update(f"{self.glue_task}/val/loss", loss)
        global_stats.update(f"{self.glue_task}/val/acc", acc)
        global_stats.update(f"{self.glue_task}/val/mcc", mcc)
        global_stats.update(f"{self.glue_task}/val/f1", f1)
        # global_stats.log_model(global_step, self, force = True, grad_scale=1)

class SentencePairClassifier(SentenceClassifier):
    def __init__(self, *args, s1_key = 'question1', s2_key = 'question2', **kwargs,):
        # print(args, kwargs)
        super().__init__(*args, **kwargs)
        self.s1_key = s1_key
        self.s2_key = s2_key

    def collate_fun(self, batch):
        # print(batch)
        batch = torch.utils.data._utils.collate.default_collate(batch)
        labels = batch["label"]
        input = self.tokenizer(
            text = batch[self.s1_key],
            text_pair = batch[self.s2_key],
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # print(input, input['input_ids'].shape)
        return input, torch.LongTensor(labels)

class STSBPairClassifier(SentencePairClassifier):
    def __init__(self, *args, s1_key = 'sentence1', s2_key = 'sentence2', **kwargs,):
        super().__init__(*args,s1_key=s1_key,s2_key=s2_key, **kwargs)
        self.criterion = nn.MSELoss()
        self.num_labels = 1
        # print(self.net.state_dict())
    
    def collate_fun(self, batch):
        # print(batch)
        batch = torch.utils.data._utils.collate.default_collate(batch)
        labels = batch["label"]
        input = self.tokenizer(
            text = batch[self.s1_key],
            text_pair = batch[self.s2_key],
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # print(input, input['input_ids'].shape)
        return input, labels.float()
    
    def pearson_and_spearman(self, labels, preds):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
        
    def on_end_val_epoch(
        self, global_step: int, *val_step_collated_outputs, key="default"
    ):
        """
        callback after validation loop ends
        """
        loss, predicted, labels = val_step_collated_outputs
        # print(labels, predicted)
        loss = loss.mean().item()
        correlations = self.pearson_and_spearman(labels, predicted)
        #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html
        global_stats.update(f"{self.glue_task}/val_{key}/loss", loss)
        global_stats.update_multi(f"{self.glue_task}/val_{key}", correlations)
    
class MNLIClassifier(SentencePairClassifier):
    def __init__(self, *args, max_batches_per_epoch = 500 , **kwargs):
        super().__init__(*args,**kwargs)
        self.max_batches_per_epoch = max_batches_per_epoch
        self.dl = None

    
    def get_val_dataloaders(self, sampler: torch.utils.data.Sampler, batch_size: int):
        return {key:torch.utils.data.DataLoader(
                ds,
                sampler = sampler(ds),
                batch_size=batch_size,
                collate_fn=self.collate_fun,
            ) for key,ds in self.data_interface.get_val_dataset().items()}

    def on_end_val_epoch(
        self, global_step: int, *val_step_collated_outputs, key="default"
    ):
        """
        callback after validation loop ends
        """
        loss, predicted, labels = val_step_collated_outputs
        # print(labels, predicted)
        loss = loss.mean().item()
        acc = accuracy_score(labels, predicted)
        global_stats.update(f"{self.glue_task}/val_{key}/loss", loss)
        global_stats.update(f"{self.glue_task}/val_{key}/acc", acc)
        # global_stats.log_model(global_step, self, force = True, grad_scale=1)

def recipe_factory(glue_task, *args, **kwargs):
    factory = {
        'cola':SentenceClassifier,
        'qqp':SentencePairClassifier,
        'rte':SentencePairClassifier,
        'qnli':SentencePairClassifier,
        'mnli':MNLIClassifier,
        'sst2':SentenceClassifier,
        'stsb':STSBPairClassifier,
        'wnli':SentencePairClassifier,
        'mrpc':SentencePairClassifier,
    }
    return factory[glue_task](*args, **kwargs)
    
from data import data_factory

def run_glue_finetune(config):

    glue_task = config['glue_task']
    print(glue_task)
    data = data_factory(glue_task)
    data.setup_datasets(glue_task)
    print(data.get_train_dataset()[:5])

    recipe = recipe_factory(glue_task, data_interface = data, **config['mi'])

    # Training code
    print(recipe)
    trainer = Trainer(
        recipe,
        TrainerArguments(
            **config["tr"], 
            writer_args=WriterInitArguments(**config["wrt"]), 
            stats_args = StatInitArguments(**config["stat"]),
            checkpointer_args= DefaultCheckpointerArguments(**config["ckp"]) if 'ckp' in config else DefaultCheckpointerArguments()
        ),
    )
    trainer.validate()
    trainer.train()

if __name__ == "__main__":
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)
    config = CustomArgParser().parse()
    print(config)
    run_glue_finetune(config)
