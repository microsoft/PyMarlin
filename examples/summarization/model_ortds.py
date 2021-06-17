from typing import List, Dict
import os

# too long import
from pymarlin.utils.stats import global_stats
from pymarlin.utils.logger import getlogger

from onnxruntime.training.ortmodule import ORTModule

from filelock import FileLock

from deepspeed_methods.deepspeed_utils import initialize_deepspeed
from deepspeed_methods.deepspeed_utils import get_core_model
from train import SummarizationBartModule

logger = getlogger(__file__)

class SummarizationBartModuleORT(SummarizationBartModule):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        #setting this here to avoid issues after wrapping
        self._pad_token_id = self.model.config.pad_token_id

        logger.info("Employing ORT, wrapping model with ORTModule")
        self.model = ORTModule(self.model)
        
    def get_core_model(self):
        return get_core_model(self.model, ort_flag=True)
    
    @property
    def pad_token_id(self):
        return self._pad_token_id

class SummarizationBartModuleDeepSpeedORT(SummarizationBartModuleORT):
    def __init__(
            self,
            *args,
            deepspeed_config='',
            deepspeed_transformer_kernel=False,
            deepspeed_ckpt_tag=None,
            deepspeed_resume_from_checkpoint=None,
            generate_kwargs={},
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        logger.info(f"Employing Deepspeed, wrapping model with Deepspeed")
        self.model, _ = initialize_deepspeed(self.model, deepspeed_config, deepspeed_transformer_kernel)
        self.deepspeed_resume_from_checkpoint = deepspeed_resume_from_checkpoint
        self.deepspeed_ckpt_tag = deepspeed_ckpt_tag
        self.DEEPSPEED_CKPT_PREFIX = "deepspeed_ckpt"

    def get_optimizers_schedulers(
            self, estimated_global_steps_per_epoch: int, epochs: int
    ):
        print(f"Deepspeed is employed, optimizer and scheduler are defined in deepspeedConfig.json file")
        return [], []

    def get_core_model(self):
        return get_core_model(self.model, ort_flag=True, deepspeed_flag=True)

    def train_step(self, global_step: int, batch, device):
        batch = batch.to(device)
        result = self.model(**batch)
        global_stats.update("lr", self.model.get_lr()[0], frequent=True)
        loss = result["loss"]

        return loss

    def get_state(self) -> Dict:
        return None

    def update_state(self, state: Dict):
        if self.deepspeed_resume_from_checkpoint is not None:

            import glob
            loading_path = os.path.join(self.deepspeed_resume_from_checkpoint, self.DEEPSPEED_CKPT_PREFIX)
            deepspeed_checkpoint_dirs = sorted(glob.glob(f"{loading_path}/*"))

            if len(deepspeed_checkpoint_dirs) > 0:
                logger.info(f"Attempting to resume from {loading_path}")
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
                logger.error(f"{loading_path} doesn't have deepspeed checkpoints, doing nothing")
