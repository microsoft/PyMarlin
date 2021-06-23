"""
Trainer module:

The `Trainer` is responsible for coordinating the model definition
(`ModuleInterface`) and the `TrainerBackend` - connecting the high-level
model recipe with the backend on which it will be trained.

This accepts a `module` implementing `ModuleInterface` that contains the
model definition, as well as the definition of train and evaluation steps,
optimizers and schedulers and any optional callbacks.

It also accepts a `TrainerBackend` defining how the training should be run
e.g. single node vs distributed training. There are `TrainerBackends` for
most common scenarios available out of the box - or alternatively a user can
provide a custom `TrainerBackend`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.auto import trange
import torch
from torch.utils.data.sampler import SequentialSampler

from pymarlin.core import trainer_backend as trn
from pymarlin.core import module_interface
from pymarlin.utils import fabrics
from pymarlin.utils import distributed
from pymarlin.utils.distributed import DistributedTrainingArguments, rank_zero_only

from pymarlin.utils.checkpointer.checkpoint_utils import AbstractCheckpointer
from pymarlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointer
from pymarlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments
from pymarlin.utils.checkpointer.checkpoint_utils import Checkpoint

from pymarlin.utils.logger.logging_utils import getlogger
from pymarlin.utils import stats
from pymarlin.utils.writer import build_writer
from pymarlin.utils.writer.base import WriterInitArguments


@dataclass
class TrainerArguments:
    """
    Trainer Arguments class.
    """
    epochs: int = 1
    use_gpu: bool = True
    train_batch_size: int = 1
    gpu_batch_size_limit: int = 512
    val_batch_size: int = 1
    max_train_steps_per_epoch: Optional[int] = None
    max_val_steps_per_epoch: Optional[int] = None
    clip_grads: bool = True
    max_grad_norm: float = 1.0
    reset_optimizers_schedulers: bool = False

    # checkpointer args
    checkpointer_args: DefaultCheckpointerArguments = DefaultCheckpointerArguments()

    # distributed training arguments
    distributed_training_args: DistributedTrainingArguments = None

    # logging
    writers: List = field(default_factory=lambda: ["stdout", "aml", "tensorboard"])
    stats_args: stats.StatInitArguments = stats.StatInitArguments()
    writer_args: WriterInitArguments = WriterInitArguments()
    disable_tqdm: bool = False
    log_level: str = "INFO"

    # backend
    backend: str = "sp"
    amp_backend_native: bool = False
    amp_backend_apex: bool = False
    amp_level_apex: str = 'O1'


class AbstractTrainer(ABC):
    """
    Abstract Trainer class.
    """

    @abstractmethod
    def train(self):
        """Run Train loop"""

    @abstractmethod
    def validate(self):
        """Run eval loop"""


class Trainer(AbstractTrainer):
    """Orchestrates model training.

    Args:
        module (ModuleInterface): Contains model definition, train and validation
            definition, optimizer and scheduler, and optional callbacks.
        args (TrainerArguments): Training hyperparameters.

        Optional keyword arguments:
        trainer_backend (TrainerBackend): How the training will be carried out.
            For example, the training is distributed and/or using AMP (automatic mixed precision).
            This can also be specified in args using the backend keyword.
            Defaults to singleprocess. Options are: sp (singleprocess), sp-amp, ddp, ddp-amp.
        checkpointer (AbstractCheckpointer): Used to handle model checkpointing.
    """

    def __init__(
            self,
            module: module_interface.ModuleInterface,
            args: TrainerArguments,
            trainer_backend: Optional[trn.TrainerBackend] = None,
            checkpointer: Optional[AbstractCheckpointer] = None
    ):
        """
        Initializes stats, writers, trainer_backend and other helper functions
        """
        self.module = module
        self.args = args
        assert not (self.args.amp_backend_native and self.args.amp_backend_apex), "Can only choose one AMP backend (native or apex), not both"
        self.trainer_backend = self._init_backend(trainer_backend)
        self.logger = getlogger(__name__, self.args.log_level)
        self._fetch_ranks()
        self._log_hparams()

        self.checkpointer = self._init_checkpointer(checkpointer)
        self.checkpointed_states = self.load_checkpoints()

        self.module.update_state(self.checkpointed_states.module_interface_state)
        self.module.to(self.device)

        opt_sc = self.module.get_optimizers_schedulers(
            self.estimated_global_steps_per_epoch, self.args.epochs
        )
        self.optimizers = opt_sc[0]
        self.schedulers = opt_sc[1]
        self.last_epoch = -1
        self.update_state(self.checkpointed_states.trainer_state)

        self._init_stats()
        self.trainer_backend.init(self._get_trainer_backend_args())
        self.trainer_backend.update_state(self.checkpointed_states.trainer_backend_state)

    def train(self):
        """ Train and validate the model"""
        for epoch in trange(
                self.last_epoch + 1, self.args.epochs, desc="Epochs", disable=self.args.disable_tqdm
        ):
            self.logger.info(f"Training epoch {epoch}")
            self.stats.update("epoch", epoch, frequent=True)
            self.module.on_begin_train_epoch(self.global_steps_finished, epoch)
            self.module.train()  # nn.module.Train
            all_outputs = self.train_epoch()

            self.logger.info("Validating")
            self.validate()

            self.module.on_end_train_epoch(self.global_steps_finished, *all_outputs)
            self.stats.log_long_stats(self.global_steps_finished)
            self.last_epoch = epoch
            self.save_checkpoint()

        # Checkpoint one final time at the end of training and save model
        self.save_checkpoint(force=True)
        self.save_model_checkpoint()

        self.module.on_end_train(self.global_steps_finished)
        self.stats.log_long_stats(self.global_steps_finished)
        self.logger.info("Finished training .. ")

    def train_epoch(self):
        all_outputs = []
        dataloader = self.module.get_train_dataloader(
            batch_size=self.train_step_batch_size, sampler=self.train_sampler
        )
        all_outputs = self.trainer_backend.train_dl(dataloader, self.module)
        return all_outputs

    def validate(self):
        """Run evaluation over multiple validation dataloaders"""
        dataloaders = self.module.get_val_dataloaders(
            batch_size=self.val_step_batch_size, sampler=self.val_sampler
        )
        self.module.eval()
        all_outputs = None
        dataloaders = (
            dataloaders if isinstance(dataloaders, dict) else {"default": dataloaders}
        )
        for key, dataloader in dataloaders.items():
            all_outputs = self.trainer_backend.validate_dl(dataloader)
            self.module.on_end_val_epoch(
                self.global_steps_finished, *all_outputs, key=key
            )
        self.stats.log_long_stats(self.global_steps_finished)
        return all_outputs

    def _fetch_ranks(self):
        """Set ranks used for distributed training.

        This method will attempt to infer the distributed training arguments from environment
        variables in the case that they are not explicitly provided already.

        Different launchers / compute fabrics have different conventions, we currently support:
            - azureml
            - torch.distributed.launch
        """

        if self.args.distributed_training_args is None:

            distributed_args = DistributedTrainingArguments()

            if fabrics.is_azureml_mpirun():
                distributed_args = distributed.fetch_ranks_from_azureml()
                distributed.set_environment_variables_for_nccl_backend()

            elif fabrics.is_torch_distributed_launch_via_environment_variables():
                distributed_args = (
                    distributed.fetch_ranks_from_torch_distributed_launch()
                )

            self.args.distributed_training_args = distributed_args
            rank_zero_only.rank = distributed_args.global_rank

    def _log_hparams(self):
        attrs = {at: getattr(self, at) for at in dir(self) if not at.startswith('__') and not callable(getattr(self, at))}
        for k, v in attrs.items():
            self.logger.info(f"{k}: {v}")

    def _init_checkpointer(self, checkpointer):
        if checkpointer is None:
            checkpointer = DefaultCheckpointer(self.args.checkpointer_args)
        return checkpointer

    def _init_backend(self, backend):
        if backend is None:
            backend = trn.build_trainer_backend(self.args.backend)
        return backend

    def save_checkpoint(self, force=False) -> None:
        """
        Checkpoint the current state of the Trainer, TrainerBackend, and ModuleInterface.

        Saves state of each object in a dictionary by calling on their get_state() methods and
        providing the states to the checkpointer's save() method.
        """
        if self.is_main_process:  # only main process should checkpoint
            checkpoint_state = Checkpoint(
                module_interface_state=self.module.get_state(),
                trainer_state=self.get_state(),
                trainer_backend_state=self.trainer_backend.get_state()
            )
            self.checkpointer.save(checkpoint_state, self.last_epoch, force)

    def save_model_checkpoint(self) -> None:
        """
        Checkpoint the current state of the ModuleInterface, used to save the final model in the
        training loop.

        Saves state of the ModuleInterface by calling on it's get_state() method and providing it
        to the checkpointer's save_model() method.
        """
        if self.is_main_process:
            self.checkpointer.save_model(self.module.get_state(), self.last_epoch)

    def load_checkpoints(self) -> Checkpoint:
        """
        Load state of Trainer, TrainerBackend, and ModuleInterface from checkpoint.

        Loading logic is determined by the checkpointer used, see DefaultCheckpointer
        for default implementation logic. If a checkpoint is loaded, all module
        states are updated.
        """
        checkpointed_states = self.checkpointer.load()
        return checkpointed_states

    def get_state(self) -> dict:
        """
        Get the current state of the Trainer for checkpointing.

        Default implementation returns epochs finished, override to include
        aditional state properties.

        Returns:
            state_dict (dict): Dictionary of variables or objects to checkpoint.
        """
        state_dict = {
            "last_epoch": self.last_epoch,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }
        return state_dict

    def update_state(self, state: dict) -> None:
        """
        Update the Trainer's state from a checkpointed state.

        Args:
            state : Output of get_state() during checkpointing.
        """
        if state:
            self.last_epoch = state["last_epoch"]
            if self.args.reset_optimizers_schedulers:
                self.logger.info("Optimizers and schedulers reset, not loaded from checkpoint")
                num_epochs_to_run = self.args.epochs - self.last_epoch - 1
                if num_epochs_to_run > 0:
                    self.optimizers, self.schedulers = self.module.get_optimizers_schedulers(
                        self.estimated_global_steps_per_epoch,
                        num_epochs_to_run
                    )
            else:
                self.logger.info("Loading optimizers and schedulers from checkpoint")
                # Assumes same number of optimizers as checkpointed, and in the same order
                for optimizer, checkpoint in zip(self.optimizers, state["optimizers"]):
                    optimizer.load_state_dict(checkpoint)
                # Assumes same number of scheduler as checkpointed, and in the same order
                for scheduler, checkpoint in zip(self.schedulers, state["schedulers"]):
                    scheduler.load_state_dict(checkpoint)

    @property
    def device(self):
        """The torch device either CPU or GPU, and the local rank.

        Note: _fetch_rank() should have already been called
        before calling device.
        """
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(
                "cuda", self.args.distributed_training_args.local_rank
            )
        else:
            device = torch.device("cpu")
        return device

    @property
    def is_distributed(self):
        return (
                self.args.distributed_training_args.world_size > 1
        )  # default world size is 1.

    @property
    def is_main_process(self):
        return self.args.distributed_training_args.global_rank == 0

    @property
    def pergpu_global_batch_size(self):
        return self.args.train_batch_size // self.args.distributed_training_args.world_size

    @property
    def gradient_accumulation(self):
        return self.pergpu_global_batch_size // self.train_step_batch_size

    @property
    def train_sampler(self) -> Optional[type]:
        return self.trainer_backend.train_sampler

    @property
    def val_sampler(self) -> Optional[type]:
        return self.trainer_backend.val_sampler

    @property
    def train_step_batch_size(self):
        """
        Returns micro batch sizes for training. Splits batch into smaller step batches such that
            1. Each step batch fits into memory
            2. step batch size are a factor of global batch size per gpu
            3. attain maxium batch size that follows 1 and 2
        """
        batch_size = self.pergpu_global_batch_size
        if batch_size > self.args.gpu_batch_size_limit:
            # Try to accomodate maximum batch size that fits in memory
            if batch_size % self.args.gpu_batch_size_limit == 0:
                batch_size = self.args.gpu_batch_size_limit
            else:
                while batch_size > self.args.gpu_batch_size_limit:
                    batch_size = batch_size // 2
        assert (
                batch_size > 0
        ), 'Train step batch size calculated is 0. Reduce GPUs or increase batch size.'
        assert (
                batch_size <= self.args.gpu_batch_size_limit
        ), "Train step batch size calculated too high. fix calculation logic"
        return batch_size

    @property
    def val_step_batch_size(self):
        return self.args.val_batch_size # this is generally larger than limit for training.

    @property
    def global_steps_finished(self):
        return self.trainer_backend.get_global_steps_completed()

    @property
    def total_steps_finished(self):
        return self.trainer_backend.get_batches_completed()

    @property
    def estimated_global_steps_per_epoch(self):
        """Estimate the number of global steps per epoch.

        Compute the maximum number of global steps as len(dataloader) // gradient_accumulation + 1.
        If max_train_steps_per_epoch is provided we return the minimum of the two.

        Note: SequentialSampler is used to get the train dataloader regardless of
        the sampler provided by trainer_backend as we only require the length of the dataloader.

        Do not change this logic without testing thorougly. There is a test case already written.

        TODO: simplify this by initiliaizing distributed environment before calling this and remove SequentialSampler.
        """
        single_process_train_dl = self.module.get_train_dataloader(
            batch_size=self.train_step_batch_size, sampler=SequentialSampler
        )

        # null check train_dl
        if single_process_train_dl is None:
            return 0

        max_global_steps_single_process = len(single_process_train_dl)
        max_global_steps = (
                max_global_steps_single_process
                // (self.args.distributed_training_args.world_size * self.gradient_accumulation)
                + 1
        )

        if self.args.max_train_steps_per_epoch:
            max_global_steps = min(
                max_global_steps, self.args.max_train_steps_per_epoch
            )
        return max_global_steps

    @property
    def stats(self):
        return stats.global_stats

    def _init_stats(self):
        if self.is_main_process:
            writers = [
                build_writer(writer, self.args.writer_args)
                if isinstance(writer, str)
                else writer
                for writer in self.args.writers
            ]
            stats.global_stats.rebuild(self.args.stats_args, writers)

    def _get_trainer_backend_args(self):
        return trn.TrainerBackendArguments(
            model=self.module,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            gradient_accumulation=self.gradient_accumulation,
            device=self.device,
            max_train_steps_per_epoch=self.args.max_train_steps_per_epoch,
            max_val_steps_per_epoch=self.args.max_val_steps_per_epoch,
            clip_grads=self.args.clip_grads,
            max_grad_norm=self.args.max_grad_norm,
            distributed_training_args=self.args.distributed_training_args,
            disable_tqdm=self.args.disable_tqdm,
            amp_backend_native=self.args.amp_backend_native,
            amp_backend_apex=self.args.amp_backend_apex,
            amp_level_apex=self.args.amp_level_apex,

        )
