"""
Checkpointer class and utility functions.
"""

import os
import re
from typing import Optional, Dict, Tuple, Callable
from abc import ABC, abstractmethod
from operator import itemgetter
from dataclasses import dataclass
import torch
from pymarlin.utils.logger.logging_utils import getlogger

@dataclass
class Checkpoint:
    """
    Checkpoint data class that holds the states for
    module_interface, trainer and trainer_backend
    """
    module_interface_state: dict = None
    trainer_state: dict = None
    trainer_backend_state: dict= None

@dataclass
class DefaultCheckpointerArguments:
    """
    Default Checkpointer Arguments.

    Args:
        checkpoint (bool): Flag indicating whether to checkpoint model when save()
            is called. Other conditions are implemented within save(), allowing this
            method to always be called within training loops and abstracting the
            checkpointing logic out of Trainer and implemented in this class.
        delete_existing_checkpoints (bool): Flag indicating whether to delete checkpoints
            under save_dir before training. New checkpoints are saved regardless.
        period (int): Period of index at which to checkpoint model. Evaluates
            index % period == 0. This function is called with index set to the
            epoch, and thus checkpoints every "period" number of epochs. The last
            epoch is always checkpointed regardless.
        save_dir (str): Path to directory where checkpoints are to be stored. Creates
            folder if it does not exist.
        model_state_save_dir (str): Path to directory where checkpointed models are to
            be stored. Creates folder if it does not exist.
        load_dir (str): Path to directory where checkpoints are to be loaded from.
            If not set, will not attempt to load a checkpoint. If load_filename
            is set, will search for this filename within the directory to load it.
            If load_filename is not set, will load the file via get_latest_file().
        load_filename (str): Filename of checkpoint to load under load_dir, overrides
            automatic loading via get_latest_file().
        file_prefix (str): Prefix of the checkpoint filename. Final filename to save
            will be {file_prefix}_{index}.{file_ext}, or in the case of saving with
            save_model(), {file_prefix}_mode_{index}.{file_ext}.
        file_ext (str): File extension for the checkpoint filename when saving and when
            searching under load_dir for loading via get_latest_file().
            When cleaning save_dir via delete_existing_checkpoints=True, only files
            with this extension are considered.
        log_level (str): Logging level for checkpointer module (Default: 'INFO').
    """
    checkpoint: bool = True
    delete_existing_checkpoints: bool = False
    period: int = 1
    save_dir: str = os.path.join(os.getcwd(), "checkpoints")
    model_state_save_dir: str = os.path.join(os.getcwd(), "checkpoints")
    load_dir: str = None
    load_filename: str = None
    file_prefix: str = "model"
    file_ext: str = "pt"
    log_level: str = "INFO"

class AbstractCheckpointer(ABC):
    """
    Abstract class for a checkpointer.

    To create a custom checkpointer, users must implement the abstract
    methods of this class and pass along an instance to ModuleInterface.
    Custom checkpointers can be used at other stages of the
    training lifecycle via callbacks.
    """

    @abstractmethod
    def save(self, checkpoint_state: Checkpoint,
             index: int, force: Optional[bool]) -> None:
        """
        Creates a checkpoint by saving a Checkpoint dataclass containing any
        relevant states.

        Args:
            checkpoint_state (Checkpoint): Checkpointed states.
            index (int): Using epoch as index is suggested.
            force (bool, optional): Saves checkpoint regardless of conditions
                if args.checkpoint is set to True. Used to always checkpoint
                models after the last epoch.
        """

    def save_model(self, model_state: Dict, index: int) -> None:
        """
        Creates a model checkpoint by saving model state.

        Args:
            model_state (Dict): Model state as provided by ModuleInterface.
            index (int): Number to use to create a unique filename.
                Using epoch as index is suggested.
        """
        pass

    @abstractmethod
    def load(self) -> Checkpoint:
        """
        Load and return a checkpointed file.

        Implements logic to load a checkpointed file as configured
        via args used when constructing the checkpointer object. Always called
        upon initialization of Trainer.

        Returns:
            Checkpoint: Checkpointed states.
        """


class DefaultCheckpointer(AbstractCheckpointer):
    """
    Default checkpointer implementation, implements AbstractCheckpointer and
    contains a few helper functions for managing checkpointed files.

    Must be initialized with DefaultCheckpointerArguments.
    """
    def __init__(self, args: DefaultCheckpointerArguments):
        """
        Initialize checkpointer and delete existing checkpointed files
        under save_dir if delete_existing_checkpoints is set to True.
        """
        self.args = args
        self.logger = getlogger(__name__, args.log_level)

        if args.delete_existing_checkpoints and os.path.exists(args.save_dir):
            self.logger.warning("Deleting checkpoints under %s with "
                                "%s extension.", args.save_dir, args.file_ext)
            for f in os.listdir(args.save_dir):
                if f.endswith(args.file_ext):
                    try:
                        os.remove(os.path.join(args.save_dir, f))
                    except FileNotFoundError as err:
                        self.logger.error('Could not delete checkpoint: %s', err)

    def save(self, checkpoint_state: Checkpoint, index: int, force=False) -> str:
        """
        Creates a checkpoint by saving a Checkpoint dataclass containing any
        relevant states as a python Dict.

        Evaluates conditions and, if met, saves a provided dataclass
        which should contain any states that users require to save as
        part of a checkpoint under args.save_dir. An additional index
        argument is required to create a unique name for the file to be
        saved. The optional force flag will disregard conditions other
        than the checkpoint flag that enables this behavior. The condition
        for saving with DefaultCheckpointer is index being a multiple
        of the args.period.

        Args:
            checkpoint_state (Checkpoint): Checkpointed states.
            index (int): Number to use to create a unique filename and
                evaluate conditions for checkpointing. Using epoch as index
                is suggested.
            force (bool, optional): Saves checkpoint regardless of conditions
                if args.checkpoint is set to True. Used to always checkpoint
                states after the last epoch.

        Returns:
            str: Path to checkpointed file.
        """
        if self.args.checkpoint and ((index % self.args.period == 0) or force):
            self.check_mk_dir(self.args.save_dir)
            path = os.path.join(self.args.save_dir,
                f"{self.args.file_prefix}_{index}.{self.args.file_ext}")
            torch.save(checkpoint_state.__dict__, path)
            self.logger.debug("Saved checkpoint %s", path)
            return path
        return None

    def save_model(self, model_state: Dict, index: int) -> str:
        """
        Checkpoints a model state, leveraging torch.save().

        Evaluates if checkpointing is enabled and if a model save directory
        has been set, and saves a provided model state. An additional index
        argument is required to create a unique name for the file to be
        saved.

        Args:
            model_state (Dict): Model state as provided by ModuleInterface.
            index (int): Number to use to create a unique filename.
                Using epoch as index is suggested.
        Returns:
            str: Path to checkpointed file.
        """
        if self.args.checkpoint and self.args.model_state_save_dir:
            self.check_mk_dir(self.args.model_state_save_dir)
            path = os.path.join(self.args.model_state_save_dir,
                f"{self.args.file_prefix}_model_{index}.{self.args.file_ext}")
            torch.save(model_state, path)
            return path
        return None

    def load(self) -> Checkpoint:
        """
        Attempt to load and return a checkpointed file leveraging torch.load().
        The checkpoined file is assumed to be created with save() and thus be
        a python Dict.

        This method is always called upon initialization of Trainer.
        Searches for and attempts to load a checkpointed file based on
        args. If no load_dir is set, returns None. If a load_dir and
        load_filename have been set, the file "load_filename" under
        load_dir is directly loaded (the filename must include extension).
        If only load_dir is set, get_latest_file() is called to seach the
        folder for the file with the largest integer (index) in its filename,
        and returns that path for loading.

        Returns:
            Checkpoint: Checkpointed states.
        """
        if self.args.load_dir:
            if self.args.load_filename:
                path = os.path.join(self.args.load_dir,
                                    self.args.load_filename)
            else:
                path = self.get_latest_file(self.args.load_dir,
                                            self.args.file_prefix,
                                            self.args.file_ext,
                                            self.logger)
            self.logger.debug('Loading checkpoint from path: %s', path)
            try:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                self.logger.debug('Checkpoint loaded')
                return Checkpoint(**checkpoint)
            except FileNotFoundError as err:
                self.logger.error(err)
                return Checkpoint()
        return Checkpoint()

    # TODO: we can make this unacessible if we find we do not need this outsider checkpointer with  model_state_save_dir
    # josleep: making this static so that we can re-use this logic in scenarios.
    @staticmethod
    def get_latest_file(load_dir: str, file_prefix: str,
                        file_ext: str = 'pt', logger = getlogger(__name__)) -> str:
        """
        Get the path to the last checkpointed file.

        Find and return the path of the file with greatest number of
        completed epochs under dirpath (recursive search) for a given file
        prefix, and optionally file extension.

        Args:
            load_dir (str): Directory under which to search for
                checkpointed files.
            file_prefix (str): Prefix to match for when searching
                for candidate files.
            file_ext (str, optional): File extension to consider
                when searching.

        Returns:
            str: Path to latest checkpointed file.
        """
        latest_path = ""
        candidate_files = []
        for dir_path, _, filenames in os.walk(load_dir):
            candidate_files.extend(
                [
                    os.path.join(dir_path, f)
                    for f in filenames
                    if f.startswith(file_prefix) and f.endswith(file_ext)
                ]
            )
        if candidate_files:
            files_and_epochs = [(fl, int(re.findall(r"\d+", fl)[-1]))
                                for fl in candidate_files]
            latest_path = max(files_and_epochs, key=itemgetter(1))[0]
        else:
            logger.info('No checkpointed files found under %s.', load_dir)
        return latest_path

    def check_mk_dir(self, dirpath: str) -> None:
        """
        Check if the path exists, and if it doesn't creates it.

        Args:
            dirpath (str): Directory under which to search for
                checkpointed files.
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        assert os.path.isdir(dirpath), "supplied checkpoint dirpath "\
            "is not a directory"


@dataclass
class BestCheckpointerArguments(DefaultCheckpointerArguments):
    """Additional arguments for checkpointer

    metric_name: name of metric where minimal is defined as best. Must be a registered buffer in module interface
    save_intermediate_checkpoints: whether to produce a checkpointer every epoch in addition to latest and best.
    load_best: whether to load best or latest checkpoint. Default behavior is to load latest.
    """
    metric_name: str = "val_perplexity"
    init_metric_val: Optional[float] = None
    criteria: Optional[Tuple[str, Callable]] = "min"
    save_intermediate_checkpoints: bool = False  # not usually necessary in practice
    load_best: bool = False  # default to load latest


class BestCheckpointer(DefaultCheckpointer):
    """
    Saves best and latest checkpoint. Best checkpoint is defined as the smallest value of a given parameter in the
    module interface. Therefore this checkpointer works by relying on the parameter defined in metric_name existing as a
    single value. By default it checks "val_perplexity" which is a registered buffer in `AbstractUserMessageReplyModule`
    that gets updated after every call to `on_end_val_epoch`.
    """
    def __init__(self, args: BestCheckpointerArguments):
        super().__init__(args)
        self.best_checkpoint_name = f"{self.args.file_prefix}_best_checkpoint.{self.args.file_ext}"
        self.latest_checkpoint_name = f"{self.args.file_prefix}_latest_checkpoint.{self.args.file_ext}"
        if self.args.criteria == 'min':
            self.criteria_func = lambda new, old: new < old
            self.best_metric = float('inf')
        elif self.args.criteria == 'max':
            self.criteria_func = lambda new, old: new > old
            self.best_metric = -float('inf')
        else:
            self.criteria_func = self.args.criteria
            self.best_metric = self.args.init_metric_value

        if self.args.init_metric_value is not None:
            self.best_metric = self.args.init_metric_value

    def save(self, checkpoint_state: Checkpoint, index: int, force=False) -> str:
        """
        Saves trainer, optimizer, and module interface state.

        Args:
            checkpoint_state: instance of `Checkpoint` which contains trainer, optimizer, and module interface state
            index: current epoch number
            force: whether to force a save even if period of checkpointing does not line up

        Returns:
            list of paths checkpoint state was saved to
        """
        paths = []
        if self.args.save_intermediate_checkpoints:
            paths.append(super().save(checkpoint_state, index, force))
        if self.args.checkpoint:
            # TODO grab this from logged metrics instead, checkpoint state is hacky
            self.logger.debug(f"Available metrics {checkpoint_state.module_interface_state.keys()}")
            metric = float(checkpoint_state.module_interface_state[self.args.metric_name])
            self.logger.info(f"epoch {index}: metric {self.args.metric_name}={metric}, best score={self.best_metric}")

            # optiionally save best
            if self.criteria_func(metric, self.best_metric):
                self.best_metric = metric
                best_path = os.path.join(self.args.save_dir, self.best_checkpoint_name)
                torch.save(checkpoint_state.__dict__, best_path)
                paths.append(best_path)

            # save latest
            latest_path = os.path.join(self.args.save_dir, self.latest_checkpoint_name)
            torch.save(checkpoint_state.__dict__, latest_path)
            paths.append(latest_path)
        return paths

    def load(self) -> Checkpoint:
        """
        Optionally loads a checkpoint from a given directory. Either loads a specified filename, the best checkpoint, or
        the latest checkpoint. Raises a `ValueError` upon failure to load checkpoint.

        Returns:
            An instance of `Checkpoint`
        """
        if self.args.load_dir:
            if self.args.load_filename:
                load_path = os.path.join(self.args.load_dir, self.args.load_filename)
            elif self.args.load_best:
                load_path = os.path.join(self.args.load_dir, self.best_checkpoint_name)
            else:
                load_path = os.path.join(self.args.load_dir, self.latest_checkpoint_name)

            # TODO how to set best metric to match loaded checkpoint?
            self.logger.debug(f"loading checkpoint from {load_path}")
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
            self.logger.debug('Checkpoint loaded')
            return Checkpoint(**checkpoint)

        return Checkpoint()
