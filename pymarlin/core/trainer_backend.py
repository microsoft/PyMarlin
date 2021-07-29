"""
Trainer Backend module:

Currently we support:
    1. SingleProcess
    2. SingleProcess Amp
    3. SingleProcess Apex-Amp
    4. DDP
    5. DDP Amp
    6. DDP Apex-Amp

These are `TrainerBackends` for most common scenarios available out of the box.
Alternatively a user can provide a custom `TrainerBackend`.
"""
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import dataclasses
from typing import Iterable, Optional, Union
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler

from pymarlin.core import module_interface
from pymarlin.utils import stats
from pymarlin.utils.distributed import (
    DistributedTrainingArguments,
    SequentialDistributedSampler,
)


try:
    from apex import amp
except ImportError:
    amp = None
from functools import wraps

def build_trainer_backend(trainer_backend_name, *args, **kwargs):
    """Factory for trainer_backends

    Args:
        trainer_backend_name (str): TrainerBackend Name. Possible choices are currently: sp, sp-amp, sp-amp-apex, ddp, ddp-amp, ddp-amp-apex
        args (sequence): TrainerBackend positional arguments
        kwargs (dict): TrainerBackend keyword arguments
    """
    factory_dict = {
        "sp": SingleProcess,
        "sp-amp": SingleProcessAmp,
        "sp-amp-apex": SingleProcessApexAmp,
        "ddp": DDPTrainerBackendFactory(SingleProcess),
        "ddp-amp": DDPTrainerBackendFactory(SingleProcessAmp),
        "ddp-amp-apex": DDPTrainerBackendFactory(SingleProcessApexAmp),
    }
    return factory_dict[trainer_backend_name](*args, **kwargs)


@dataclasses.dataclass
class TrainerBackendArguments:
    """
    Trainer Backend Arguments dataclass.
    """
    model: module_interface.ModuleInterface
    device: Union[torch.device, str, int]
    max_train_steps_per_epoch: Optional[int]
    max_val_steps_per_epoch: Optional[int]
    distributed_training_args: DistributedTrainingArguments
    optimizers: Iterable[torch.optim.Optimizer]
    schedulers: Optional[Iterable[torch.optim.lr_scheduler._LRScheduler]] = None # pylint: disable=protected-access
    gradient_accumulation: int = 1
    clip_grads: bool = True
    max_grad_norm: float = 1.0
    disable_tqdm: bool = False
    enable_amp: bool = True
    amp_backend_native: bool = False
    amp_backend_apex: bool = False
    amp_level_apex: str = 'O1'


class TrainerBackend(ABC):
    """
    Trainer Backend abstract class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def init(self, args: TrainerBackendArguments):
        pass

    @abstractmethod
    def train_dl(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate_dl(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_batches_completed(self):
        pass

    @abstractmethod
    def get_global_steps_completed(self):
        pass

    @property
    @abstractmethod
    def train_sampler(self):
        return RandomSampler

    @property
    @abstractmethod
    def val_sampler(self):
        return SequentialSampler

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def update_state(self, state):
        pass


class OutputCollector():
    """
    Responsible for collecting step outputs and stores them in memory across each call.
    Concatinates tensors from all steps across first dimension.
    """

    def __init__(self):
        self.all_outputs = []  # keeps track of all outputs

    def reset(self):
        self.all_outputs = []

    def collect(self, outputs: Union[torch.Tensor, Iterable[torch.Tensor]]):
        """
        Coalesces train_step and val_step outputs.
        all tensors concatenated across dimension 0
        if input is a torch.Tensor of dimension batch_size * x* y .., all_outputs will be List[torch.Tensor of dimension total_samples_till_now *x *y]
        if input is a torch.Tensor of dimension 1 * 1, all_outputs will List[torch.Tensor of dimension total_samples_till_now * 1]
        if input is List[torch.Tensor], all_outputs will be List[torch.Tensor] - all tensors concatenated across dimension 0

        Args:
            outputs (Union[torch.Tensor, Iterable[torch.Tensor]]): train_step , val_step outputs
        """
        # convert to iterable if output is a single tensor
        outputs_iter = [outputs] if (isinstance(outputs, torch.Tensor) or not isinstance(outputs, Iterable)) else outputs
        for i, output in enumerate(outputs_iter):
            # detach from graph and move to cpu
            if isinstance(output, torch.Tensor):
                tensor = output.detach().to('cpu')
                self._append_tensor(tensor, i)
            else:
                self._append_nontensor(output, i)

    def _append_tensor(self, tensor, index):
        # check for first time append
        if len(self.all_outputs) < index + 1:
            self.all_outputs.append(tensor)
        else:
            self.all_outputs[index] = self._safe_cat(self.all_outputs[index], tensor)

    def _append_nontensor(self, output, index):
        # check for first time append
        if len(self.all_outputs) < index + 1:
            self.all_outputs.append([])
        self.all_outputs[index].append(output)

    # pylint: disable=not-callable
    @staticmethod
    def _safe_cat(a: torch.Tensor, b: torch.Tensor):
        """Safely apply torch.cat.

        Handles the case where tensors have dimension 0 by
        unsqueezing.
        """
        a = torch.tensor([a]) if a.dim() == 0 else a
        b = torch.tensor([b]) if b.dim() == 0 else b
        return torch.cat([a, b], dim=0)

class SingleProcess(TrainerBackend):
    """Single Process Trainer Backend"""

    # pylint: disable=super-init-not-called
    def __init__(self):
        """
        Single process trainer_backend
        """
        self.global_step_completed = 0
        self.batches_completed = 0
        self.distributed = False

    @property
    def stats(self):
        return stats.global_stats

    def init(self, args: TrainerBackendArguments):
        self.args = args
        self.model = self.args.model
        if not self.distributed:
            assert self.args.distributed_training_args.world_size == 1 \
                , 'World size > 1 . Decorate with DDPTrainerBackend'

    def get_batches_completed(self):
        return self.batches_completed

    def get_global_steps_completed(self):
        return self.global_step_completed

    def train_dl(self, dataloader, callback: module_interface.CallbackInterface):

        epoch_collector = OutputCollector()
        global_step_collector = OutputCollector()
        self.global_step_this_epoch = 0
        # can pass certain stuff as argument instead of passing the entire train module.
        # But will this hinder inheritence as different trainer_backends will need different stuff from train module
        with tqdm(dataloader, unit="batch", disable=self.args.disable_tqdm) as tbatch:
            for _, batch in enumerate(tbatch):
                if (
                        self.args.max_train_steps_per_epoch
                        and self.global_step_this_epoch
                        >= self.args.max_train_steps_per_epoch
                ):
                    break

                tbatch.set_description(f"Training {self.args.distributed_training_args.global_rank}")
                outputs = self._forward_backward(callback, batch)

                # collect
                epoch_collector.collect(outputs)
                global_step_collector.collect(outputs)

                unscaled_loss = outputs[0].item()
                tbatch.set_postfix(
                    loss=unscaled_loss
                )  # move progress bar to logger later

                self.batches_completed += 1

                if self.batches_completed % self.args.gradient_accumulation == 0:
                    # write global step mean loss to stats
                    self.process_global_step(global_step_collector, callback)

        return epoch_collector.all_outputs

    def _forward_backward(self, callback, batch):
        # forward
        outputs = self.model.forward(
            stage=module_interface.Stage.TRAIN,
            batch=batch,
            device=self.args.device,
            global_step=self.global_step_completed + 1,
        )
        # assume iterable if first return type is not a list
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        # Reduce loss by ga factor since gradients are summed. Will result in large gradients otherwise
        loss = outputs[0] / self.args.gradient_accumulation
        # backward. This will keep on accumulating gradients
        loss.backward()
        callback.on_end_backward(self.global_step_completed, loss)
        return outputs

    def process_global_step(self, global_step_collector, callback):
        """Clip gradients and call optimizer + scheduler
        """
        global_step_outputs = global_step_collector.all_outputs
        global_step_mean_loss = (
            global_step_outputs[0].mean().item()
        )
        global_step_collector.reset()
        self.stats.update("loss", global_step_mean_loss, frequent=True)

        # gradient clipping. There should be different clippings for multiple optimizers though
        self._clip_gradients()

        # step
        self.optimize(self.args.optimizers, self.args.schedulers)

        self.global_step_completed += 1
        self.global_step_this_epoch += 1

        callback.on_end_train_step(self.global_step_completed, *global_step_outputs)
        self.stats.log_stats(self.global_step_completed)

    def _clip_gradients(self):
        if self.args.clip_grads:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )

    def validate_dl(self, dataloader):
        collector = OutputCollector()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Validation {self.args.distributed_training_args.global_rank}", disable=self.args.disable_tqdm)):
            if (
                    self.args.max_val_steps_per_epoch
                    and i >= self.args.max_val_steps_per_epoch
            ):
                break
            with torch.no_grad():
                outputs = self.model.forward(
                    batch=batch,
                    stage=module_interface.Stage.VAL,
                    device=self.args.device,
                    global_step=self.global_step_completed,
                )

            collector.collect(outputs)
        return collector.all_outputs

    def optimize(self, optimizers, schedulers):
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        if schedulers:
            for scheduler in schedulers:
                scheduler.step()

    @property
    def train_sampler(self):
        return RandomSampler

    @property
    def val_sampler(self):
        return SequentialSampler

    def get_state(self) -> dict:
        """
        Get the current state of the trainer_backend, used for checkpointing.

        Returns:
            state_dict (dict): Dictionary of variables or objects to checkpoint.
        """
        state_dict = {
            "global_step_completed": self.global_step_completed,
            "batches_completed": self.batches_completed,
        }
        return state_dict

    def update_state(self, state) -> None:
        """
        Update the trainer_backend from a checkpointed state.

        Args:
            state (dict) : Output of get_state() during checkpointing
        """
        if state:
            self.global_step_completed = state["global_step_completed"]
            self.batches_completed = state["batches_completed"]


# TODO: Merge SingleProcess and SingleProcessAmp after convergence test
# jsleep: was this convergence test run and should this be merged?
class SingleProcessAmp(SingleProcess):
    """ SingleProcess + Native PyTorch AMP Trainer Backend"""
    def __init__(self, enable_amp=True, **superclass_kwargs):
        super().__init__(**superclass_kwargs)
        self.enable_amp = enable_amp
        self.amp_handle = None

    def init(self, args: TrainerBackendArguments):
        super().init(args)
        self.scaler = GradScaler(init_scale=4096, enabled=self.enable_amp)

    def train_dl(self, dataloader, callback: module_interface.CallbackInterface):

        epoch_collector = OutputCollector()
        global_step_collector = OutputCollector()
        self.global_step_this_epoch = 0
        # can pass certain stuff as argument instead of passing the entire train module.
        # But will this hinder inheritence as different trainer_backends will need different stuff from train module
        with tqdm(dataloader, unit="batch", disable=self.args.disable_tqdm) as tbatch:
            for _, batch in enumerate(tbatch):
                if (
                        self.args.max_train_steps_per_epoch
                        and self.global_step_this_epoch
                        >= self.args.max_train_steps_per_epoch
                ):
                    break

                tbatch.set_description(f"Training {self.args.distributed_training_args.global_rank}")
                outputs = self._forward_backward(callback, batch)

                # collect
                epoch_collector.collect(outputs)
                global_step_collector.collect(outputs)

                unscaled_loss = outputs[0].item()  # even though gradients are scaled, loss will be unscaled
                tbatch.set_postfix(
                        loss=unscaled_loss,
                        global_batch = self.global_step_completed + 1)

                self.batches_completed += 1

                if self.batches_completed % self.args.gradient_accumulation == 0:
                    # write global step mean loss to stats
                    self.process_global_step(global_step_collector, callback)
        return epoch_collector.all_outputs

    def _forward_backward(self, callback, batch):
        # forward
        outputs = self._forward(batch, module_interface.Stage.TRAIN, self.global_step_completed + 1)
        # assume iterable if first return type is not a list
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        # Reduce loss by ga factor since gradients are summed. Will result in large gradients otherwise
        loss = outputs[0] / self.args.gradient_accumulation
        # backward. This will keep on accumulating gradients
        self._backward(loss)
        callback.on_end_backward(self.global_step_completed, loss)
        return outputs

    def _forward(self, batch, stage, global_step):
        with autocast(enabled=self.enable_amp):
            outputs = self.model.forward(
                stage=stage,
                batch=batch,
                device=self.args.device,
                global_step=global_step,
            )
        return outputs

    def _backward(self, loss):
        self.scaler.scale(
            loss
        ).backward()  # loss scaling. Gradients will be scaled until scale._unscale or scaler.step is called

    def validate_dl(self, dataloader):
        collector = OutputCollector()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Validation {self.args.distributed_training_args.global_rank}", disable=self.args.disable_tqdm)):
            if (
                    self.args.max_val_steps_per_epoch
                    and i >= self.args.max_val_steps_per_epoch
            ):
                break
            with torch.no_grad():
                outputs = self._forward(batch, module_interface.Stage.VAL, self.global_step_completed)

            collector.collect(outputs)
        return collector.all_outputs

    def _clip_gradients(self):
        # unscale params of each optimizer before gradient clipping
        for optimizer in self.args.optimizers:
            self.scaler.unscale_(optimizer)
        if self.args.clip_grads:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )

    def optimize(self, optimizers, schedulers):
        for optimizer in optimizers:
            self.scaler.step(optimizer)
            optimizer.zero_grad()
        self.scaler.update()

        if schedulers:
            for scheduler in schedulers:
                scheduler.step()


class SingleProcessApexAmp(SingleProcessAmp):
    """ SingleProcess + Apex AMP Trainer Backend """

    def __init__(self, enable_amp=True, **superclass_kwargs):
        super().__init__(**superclass_kwargs)
        self.enable_amp = enable_amp
        self.amp_handle = None

    def init(self, args: TrainerBackendArguments):
        super().init(args)

        if self.enable_amp:
            assert amp is not None, "apex amp cannot be found, please check if apex installed properly"
            self.amp_handle = amp
            self.model, self.args.optimizers = self.amp_handle.initialize(self.model, list(self.args.optimizers), opt_level=self.args.amp_level_apex, loss_scale="dynamic")

    def _forward(self, batch, stage, global_step):
        outputs = self.model.forward(
            stage=stage,
            batch=batch,
            device=self.args.device,
            global_step=global_step,
        )
        return outputs

    def _backward(self, loss):
        with self.amp_handle.scale_loss(loss, self.args.optimizers) as scaled_loss:
            scaled_loss.backward()

    def get_state(self) -> dict:
        state_dict = {
            "global_step_completed": self.global_step_completed,
            "batches_completed": self.batches_completed,
            "amp_state": self.amp_handle.state_dict()
        }
        return state_dict

    def update_state(self, state) -> None:
        if state:
            self.global_step_completed = state["global_step_completed"]
            self.batches_completed = state["batches_completed"]
            self.amp_handle.load_state_dict(state["amp_state"])


    def _clip_gradients(self):
        if self.args.clip_grads:
            for optimizer in self.args.optimizers:
                torch.nn.utils.clip_grad_norm_(self.amp_handle.master_params(optimizer), self.args.max_grad_norm)

    def optimize(self, optimizers, schedulers):
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        if schedulers:
            for scheduler in schedulers:
                scheduler.step()


class AbstractTrainerBackendDecorator(TrainerBackend):
    """Abstract class implementing the decorator design pattern."""

    # pylint: disable=super-init-not-called
    def __init__(self, trainer_backend):
        self.trainer_backend = trainer_backend

    def init(self, args: TrainerBackendArguments):
        self.trainer_backend.init(args)

    def train_dl(self, *args, **kwargs):
        return self.trainer_backend.train_dl(*args, **kwargs)

    def validate_dl(self, *args, **kwargs):
        return self.trainer_backend.validate_dl(*args, **kwargs)

    def get_batches_completed(self):
        return self.trainer_backend.get_batches_completed()

    def get_global_steps_completed(self):
        return self.trainer_backend.get_global_steps_completed()

    @property
    def train_sampler(self):
        return self.trainer_backend.train_sampler

    @property
    def val_sampler(self):
        return self.trainer_backend.val_sampler

    def get_state(self):
        return self.trainer_backend.get_state()

    def update_state(self, state):
        return self.trainer_backend.update_state(state)


class DDPTrainerBackend(AbstractTrainerBackendDecorator):
    """Distributed Data Parallel TrainerBackend.

    Wraps ModuleInterface model with DistributedDataParallel which handles
    gradient averaging across processes.

    .. note: Assumes initiailized model parameters are consistent across
        processes - e.g. by using same random seed in each process at
        point of model initialization.
    """
    # pylint: disable=super-init-not-called
    def __init__(self, trainer_backend, gather_frequency: Optional[int] = None):
        self.trainer_backend = trainer_backend
        self.gather_frequency = gather_frequency
        self.trainer_backend.distributed = True
        self.trainer_backend._forward_backward = self._decorate_forward_backward(self.trainer_backend._forward_backward)

    def init(self, args: TrainerBackendArguments):
        # unpack trainer_backend arguments
        self.args = args
        self.distributed_training_args = args.distributed_training_args

        # Need to initiate the distributed env and set default devices before initializing APEX AMP, otherwise may hit CUDA memory error
        self.setup_distributed_env()

        super().init(args)

        # wrapping up model
        self.trainer_backend.model = DistributedDataParallel(
            self.args.model,
            device_ids=[self.args.device],
            output_device=self.args.device,
            find_unused_parameters=True,
        )

    def setup_distributed_env(self):
        """Setup the process group for distributed training."""

        torch.distributed.init_process_group(
            backend=self.distributed_training_args.backend,
            init_method=self.distributed_training_args.init_method,
            rank=self.distributed_training_args.global_rank,
            world_size=self.distributed_training_args.world_size,
        )

        torch.cuda.set_device(self.distributed_training_args.local_rank)

    def cleanup(self):
        """Destroy the process group used for distributed training."""
        torch.distributed.destroy_process_group()

    def train_dl(self, dataloader, callback):
        all_outputs = self.trainer_backend.train_dl(dataloader, callback)
        coalesced_outputs = self._coalesce_outputs(all_outputs)
        return coalesced_outputs

    def validate_dl(self, dataloader):
        all_outputs = self.trainer_backend.validate_dl(dataloader)
        coalesced_outputs = self._coalesce_outputs(all_outputs)
        return coalesced_outputs

    def _coalesce_outputs(self, all_outputs):
        """Use all_gather to coalesce outputs across different processes.

        Gathers all tensors in all_outputs across different processes. Tensors
        are moved to CUDA and gathered in chunks of size `self.gather_frequency`.

        .. note: Use of all_gather is sub-optimal here - we only need to
            gather on rank 0. At this time (2020/12/17) NCCL backend does
            not support gather.
        """
        coalesced_outputs = []
        for x in all_outputs:

            if isinstance(x, torch.Tensor):
                gathered_x = self.gather_tensors_on_cpu(x)
                coalesced_outputs.append(gathered_x)
            else:
                msg = f"Some model outputs are not tensors (detected {type(x)})" \
                    ", and therefore will not be gathered between processes."
                warnings.warn(msg)

        return coalesced_outputs

    def _decorate_forward_backward(self, fwbw):
        # Decorates single process backward to enable or disable all reduce
        # disables all reduce if optimizer is not syncing.
        # Significant speed improvement.
        @wraps(fwbw)
        def new_fw_bw(*args, **kwargs):
            # self.batches_completed is not incremented yet.
            if ((self.trainer_backend.batches_completed+1) % self.args.gradient_accumulation) == 0:
                result = fwbw(*args, **kwargs)
            else:
                with self.trainer_backend.model.no_sync():
                    result = fwbw(*args, **kwargs)
            return result
        return new_fw_bw

    def gather_tensors_on_cpu(self, x: torch.tensor):
        """Gather tensors and move to cpu at configurable frequency.

        Move tensor to CUDA device, apply all-gather and move back to CPU.
        If `distributed_training_args.gather_frequency` is set,  tensors are
        moved to CUDA in chunks of that size.

        Args:
            x (torch.tensor): To be gathered.

        Return:
            Gathered tensor on the cpu.
        """
        n_samples = len(x)
        self._set_gather_frequency(n_samples)

        gathered = []
        n_chunks = n_samples // self.gather_frequency + 1
        for i in range(n_chunks):
            # get chunk on cpu
            chunk_cpu = x[i * self.gather_frequency: (i + 1) * self.gather_frequency]

            # move chunk to GPU
            chunk_gpu = chunk_cpu.to(self.args.device)

            # gather tensors
            gathered_chunks = [
                torch.zeros_like(chunk_gpu)
                for _ in range(self.distributed_training_args.world_size)
            ]
            torch.distributed.barrier()
            torch.distributed.all_gather(tensor_list=gathered_chunks, tensor=chunk_gpu)

            # move to cpu
            gathered_chunks_cpu = [gathered_chunk.to("cpu") for gathered_chunk in gathered_chunks]

            gathered.extend(gathered_chunks_cpu)

        # flatten gathered chunks
        flattened_gathered = torch.cat(gathered, dim=0)

        return flattened_gathered

    def _set_gather_frequency(self, n_samples):
        if self.gather_frequency is None:
            self.gather_frequency = n_samples

    @property
    def train_sampler(self):
        return DistributedSampler

    @property
    def val_sampler(self):
        return SequentialDistributedSampler

def DDPTrainerBackendFactory(trainer_backend_cls): # pylint: disable=invalid-name
    def create(*args, gather_frequency: Optional[int] = None, **kwargs):
        # pull out args to DDPTrainerBackend if needed here.
        return DDPTrainerBackend(trainer_backend_cls(*args, **kwargs), gather_frequency=gather_frequency)

    return create
