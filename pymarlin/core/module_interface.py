"""
Module Interface module:

This module contains the abstract classes CallbackInterface and
ModuleInterface that can provide everything necessary for model
training. Users should implement these abstract classes in their
Scenarios.
"""
from abc import ABC, abstractmethod
import enum
from typing import Iterable, Tuple, Union, Dict
import torch


class Stage(enum.Enum):
    """Stages: train, val, test"""
    TRAIN = 1
    VAL = 2
    TEST = 3

class CallbackInterface(ABC):
    """A callback class used to add scenario specific outputs/logging/debugging during training.
    """
    def on_begin_train_epoch(self, global_step: int, epoch: int):
        """Hook before training epoch (before model forward).

        Args:
            global_step (int): [description]
            epoch (int): Current training epoch
        """

    def on_end_train_step(self, global_step:int, *train_step_collated_outputs):
        """Runs after end of a global training step.

        Args:
            global_step (int): current global step
            train_step_collated_outputs (list): all train step outputs in a list.
                If train_step returns loss, logits train_step_collated_outputs will have [loss_collated, logits_collated]
        """

    def on_end_train_epoch(self, global_step:int, *train_step_collated_outputs):
        """
        Hook after training epoch.

        Args:
            global_step (int): [description]
            train_step_collated_outputs (list): all train step outputs in a list.
                If train_step returns loss, logits train_step_collated_outputs will have [loss_collated, logits_collated]
        """

    def on_end_backward(self, global_step:int, loss_tensor):
        """Hook after each backward

        Args:
            global_step (int): [description]
            loss_tensor(torch.Tensor): Undetached loss tensor
        """

    def on_end_val_epoch(self, global_step:int, *val_step_collated_outputs, key="default"):
        """Update value at end of end of end of variable

        Args:
            global_step (int): [description]
            val_step_collated_outputs : all val step outputs in a list.
                If val_step returns loss, logits train_step_collated_outputs will have [loss_collated, logits_collated]
            key (str, optional): The id of the validation dataloader.
                Defaults to "default".
        """

    def on_end_train(self, global_step:int):
        """Hook after training finishes

        Args:
            global_step (int): [description]
        """


class ModuleInterface(torch.nn.Module, CallbackInterface):
    """Interface for PyTorch modules.

    This interface contains model architecture in the form of a PyTorch
    `nn.Module` together with optimizers and schedules, train and validation
    step recipes and any callbacks.

    Note: The forward function is overridden.

    Note: Users are encouraged to override the `train_step` and `val_step`
    methods.
    """
    @abstractmethod
    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
        ) -> Tuple[Iterable[torch.optim.Optimizer], Iterable]:
        """
        Returns a list of optimizers and schedulers
        that are used to instantiate the optimizers .

        Returns:
            Tuple[Iterable[torch.optim.Optimizer], Iterable]:
                list of optimizers and list of schedulers
        """

    @abstractmethod
    def get_train_dataloader(
        self, sampler:type, batch_size:int
        ) -> torch.utils.data.DataLoader:
        """
        Returns a dataloader for the training loop .
        Called every epoch.

        Args:
            sampler (type): data sampler type which is a derived class of torch.utils.data.Sampler
            Create concrete sampler object before creating dataloader.
            batch_size (int): batch size per step per device

        Returns:
            torch.utils.data.DataLoader: Training dataloader
        Example:
            train_ds = self.data.get_train_dataset()
            dl = DataLoader(train_ds, batch_size = batch_size, collate_fn= self.collate_fin, sampler = sampler(train_ds))
            return dl
        """

    @abstractmethod
    def get_val_dataloaders(
        self, sampler:torch.utils.data.Sampler, batch_size : int
    ) -> Union[
        Dict[str, torch.utils.data.DataLoader],
        torch.utils.data.DataLoader
    ]:
        """
        Returns dataloader(s) for validation loop .
        Supports multiple dataloaders based on key value.
        Keys will be passed in the callback functions.
        Called every epoch .

        Args:
            sampler (type): data sampler type which is a derived class of torch.utils.data.Sampler
            Create concrete sampler object before creating dataloader.
            batch_size (int): validation batch size per step per device

        Returns:
            Union[ Dict[str, torch.utils.data.DataLoader],
            torch.utils.data.DataLoader ]:
            A single dataloader or a dictionary of dataloaders
            with key as the data id and value as dataloader
        """

    def get_test_dataloaders(self, sampler, batch_size):
        """
        Returns test dataloaders

        Args:
            sampler ([type]): [description]
            batch_size ([type]): [description]
        """
        pass

    def forward(
        self,
        stage: Stage,
        global_step: int,
        batch,
        device: Union[torch.device, str, int],
    ):
        """
        torch.nn.Module's forward() function.
        Overridden to call train_step() or val_step() based on stage .

        Args:
            stage (Stage): trian/val/test
            global_step (int): current global step
            batch ([type]): output of dataloader step
            device (Union[torch.device, str, int]): device

        Raises:
            AttributeError: if stage is different than train, val, test
        """
        if stage == Stage.TRAIN:
            return self.train_step(
                batch = batch, device = device, global_step = global_step)
        elif stage == Stage.VAL:
            return self.val_step(
                batch = batch, device = device, global_step = global_step)
        elif stage == Stage.TEST:
            return self.test_step(
                batch = batch, device = device, global_step = global_step)
        else:
            raise AttributeError("Stage not supported")

    @abstractmethod
    def train_step(
        self, global_step: int, batch, device : Union[torch.device, str, int]
        ) -> Union[torch.Tensor, Tuple]:
        """
        Train a single train step .
        Batch should be moved to device before any operation.

        Args:
            global_step (int): [description]
            batch ([type]): output of train dataloader step
            device (Union[torch.device, str, int]): device

        Returns:
            Union[torch.Tensor, Iterable[torch.Tensor]]:
                The first return value must be the loss tensor.
                Can return more than one values in output. All outputs must be tensors
                Callbacks will collate all outputs.
        """



    @abstractmethod
    def val_step(self, global_step: int, batch, device) -> Tuple:
        """
        Runs a single Validation step .

        Args:
            global_step (int): [description]
            batch ([type]): [description]
            device ([type]): [description]
        Returns:
            Union[torch.Tensor, Iterable[torch.Tensor]]: values that need to be collected - loss, logits etc.
            All outputs must be tensors
        """

    def test_step(self, global_step: int, batch, device):
        """
        Runs a single test step .

        Args:
            global_step (int): [description]
            batch ([type]): [description]
            device ([type]): [description]
        """

    def get_state(self):
        """
        Get the current state of the module, used for checkpointing.

        Returns:
            Dict: Dictionary of variables or objects to checkpoint.
        """
        state_dict = self.state_dict()
        return state_dict

    def update_state(self, state: Dict):
        """
        Update the module from a checkpointed state.

        Args:
            state (Dict): Output of get_state() during checkpointing.
        """
        if state:
            self.load_state_dict(state)
