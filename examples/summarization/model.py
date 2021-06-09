from typing import List, Dict
import os

# too long import
from pymarlin.utils.stats import global_stats

from data import SummarizationData
from onnxruntime.training.ortmodule import ORTModule

from filelock import FileLock

from deepspeed.deepspeed_utils import initialize_deepspeed
from deepspeed.deepspeed_utils import get_core_model

from train import SummarizationBartModule

try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class SummarizationBartModule_ds_ort(SummarizationBartModule):
    def __init__(
            self,
            data: SummarizationData,
            max_length_encoder=128,
            max_length_decoder=128,
            max_lr=2e-5,
            ort=False,
            deepspeed=False,
            deepspeed_config='',
            deepspeed_transformer_kernel=False,
            generate_kwargs={}
    ):
        super().__init__(data, max_length_encoder, max_length_decoder, max_lr, generate_kwargs)

        if ort:
            print(f"Employing ORT, wrapping model with ORTModule")
            self.model = ORTModule(self.model)
        if deepspeed:
            print(f"Employing Deepspeed, wrapping model with Deepspeed")
            self.model, _ = initialize_deepspeed(self.model, deepspeed_config, deepspeed_transformer_kernel)

        self.ort = ort
        self.deepspeed = deepspeed
        self.deepspeed_resume_from_checkpoint = None
        self.deepspeed_ckpt_tag = None
        self.DEEPSPEED_CKPT_PREFIX = "deepspeed_ckpt"

    def get_optimizers_schedulers(
            self, estimated_global_steps_per_epoch: int, epochs: int
    ):
        if self.deepspeed:
            print(f"Deepspeed is employed, optimizer and scheduler are defined in deepspeedConfig.json file")
            return [], []
        else:
            return super().get_optimizers_schedulers(estimated_global_steps_per_epoch, epochs)

    def train_step(self, global_step: int, batch, device):
        batch = batch.to(device)

        result = self.model(**batch)
        if self.deepspeed:
            global_stats.update("lr", self.model.get_lr()[0], frequent=True)
        else:
            global_stats.update("lr", self.schedulers.get_last_lr()[0], frequent=True)

        loss = result["loss"]

        return loss

    def val_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        module = get_core_model(self.model, deepspeed_flag=self.deepspeed, ort_flag=self.ort)

        summaries = module.generate(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        )
        preds = self.tokenizer.batch_decode(
            summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        labels = batch.labels
        labels[labels[:, :] == -100] = self.pad_token_id
        refs = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return preds, refs

    def get_state(self) -> Dict:
        if self.deepspeed:
            return None
        else:
            return super().get_state()

    def update_state(self, state: Dict):
        if self.deepspeed and (self.deepspeed_resume_from_checkpoint is not None):

            import glob
            loading_path = os.path.join(self.deepspeed_resume_from_checkpoint, self.DEEPSPEED_CKPT_PREFIX)
            deepspeed_checkpoint_dirs = sorted(glob.glob(f"{loading_path}/*"))

            if len(deepspeed_checkpoint_dirs) > 0:
                print(f"Attempting to resume from {loading_path}")
                # this magically updates self.optimizer and self.lr_scheduler
                load_path, _ = self.model.load_checkpoint(
                    loading_path,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    tag=self.deepspeed_ckpt_tag,
                )
                if load_path is None:
                    raise ValueError(f"[deepspeed] failed to resume from checkpoint {self.deepspeed_resume_from_checkpoint}")
            else:
                print(f"{loading_path} doesn't have deepspeed checkpoints, doing nothing")

        else:
            super().update_state(state)
