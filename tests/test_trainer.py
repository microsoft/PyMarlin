"""Module to test train module class"""
import os
import unittest
from unittest import mock
from marlin.core.trainer import Trainer, TrainerArguments
from marlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments


## Wrapper class to allow Mock to be pickable for checkpoint test
class PickableMock(mock.MagicMock):
    def __reduce__(self):
        return (mock.MagicMock, ())


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_trainer_backend, self.mock_module = self._create_mocks()

        ckpt_args = DefaultCheckpointerArguments(
            checkpoint=False,
            delete_existing_checkpoints=False,
            period=1,
            save_dir=os.path.join("outputs", "save_dir"),
            model_state_save_dir=os.path.join("outputs", "model_save_dir"),
            load_dir=os.path.join("outputs", "load_dir"),
            file_prefix="test",
            file_ext="tar",
        )
        self.args = TrainerArguments(
            train_batch_size=6,
            gpu_batch_size_limit=1000,  # assume high gpu memory
            val_batch_size=8,
            epochs=3,
            checkpointer_args=ckpt_args,
        )
        self.trainer = Trainer(
            trainer_backend=self.mock_trainer_backend, module=self.mock_module, args=self.args
        )

    def tearDown(self):
        self._clear_folder(self.args.checkpointer_args.save_dir, self.args.checkpointer_args.file_ext)
        self._clear_folder(self.args.checkpointer_args.model_state_save_dir, self.args.checkpointer_args.file_ext)

    def _clear_folder(self, folder, extension):
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(extension):
                    os.remove(os.path.join(folder, f))

    def _create_mocks(self):
        pickable_mock_trainer_backend = PickableMock()
        pickable_mock_module = PickableMock()
        pickable_mock_module.get_optimizers_schedulers.return_value = [
            PickableMock()
        ] * 2
        pickable_mock_module.get_state.return_value = dict()
        pickable_mock_trainer_backend.get_state.return_value = dict()
        return pickable_mock_trainer_backend, pickable_mock_module

    def _count_files(self, folder, extension):
        n_files = 0
        for f in os.listdir(folder):
            if f.endswith(extension):
                n_files += 1
        return n_files

    def test_init(self):
        self.mock_trainer_backend.init.assert_called()
        assert self.trainer.module == self.mock_module
        assert self.trainer.trainer_backend == self.mock_trainer_backend

    def test_properties(self):

        # High GPU memory
        assert self.trainer.train_step_batch_size == 6
        assert self.trainer.gradient_accumulation == 1
        assert self.trainer.val_step_batch_size == 8

    def test_gradient_accumulation(self):
        # Low GPU memory
        self.args.gpu_batch_size_limit = 3

        trainer = Trainer(
            trainer_backend=self.mock_trainer_backend, module=self.mock_module, args=self.args
        )
        assert trainer.train_step_batch_size == 3
        assert trainer.gradient_accumulation == 2
        assert trainer.val_step_batch_size == 3

    def test_train(self):
        self.trainer.train()
        # trainer_backend
        assert self.mock_trainer_backend.train_dl.call_count == 3
        assert self.mock_trainer_backend.validate_dl.call_count == 3

        # module
        assert (
            self.mock_module.get_train_dataloader.call_count == 3 + 1 + 2 
        )  # 3 during training, 1 during scheduler initialization, 2 during logging hparams
        assert self.mock_module.get_val_dataloaders.call_count == 3

        # callbacks
        assert self.mock_module.on_end_train_epoch.call_count == 3
        assert self.mock_module.on_end_val_epoch.call_count == 3
        assert self.mock_module.on_end_train.call_count == 1

    def test_validate(self):
        self.trainer.validate()
        assert self.mock_module.get_val_dataloaders.call_count == 1
        assert self.mock_trainer_backend.validate_dl.call_count == 1
        assert self.mock_module.on_end_val_epoch.call_count == 1
        assert self.mock_module.on_end_train.call_count == 0

    def test_estimated_steps(self):
        self.trainer.args.distributed_training_args.world_size = 2
        self.trainer.args.train_batch_size = 4 #global batch
        self.trainer.args.gpu_batch_size_limit = 1

        self.trainer.module.get_train_dataloader.return_value = [0]*100 # each batch is of size 1. data size  = 100

        estimated_global_steps = 100/4 #same for all gpus
        assert self.trainer.estimated_global_steps_per_epoch > estimated_global_steps
        assert estimated_global_steps + 1 >= self.trainer.estimated_global_steps_per_epoch 

    def test_save_checkpoint(self):
        # Checkpoint 3 epochs under save_dir
        self.args.checkpointer_args.checkpoint = True
        self.args.checkpointer_args.delete_existing_checkpoints = False
        self.args.epochs = 3
        self.trainer = Trainer(
            trainer_backend=self.mock_trainer_backend, module=self.mock_module, args=self.args
        )
        self.trainer.train()
        assert self._count_files(self.args.checkpointer_args.save_dir, self.args.checkpointer_args.file_ext) == 3
        assert self._count_files(self.args.checkpointer_args.model_state_save_dir, self.args.checkpointer_args.file_ext) == 1

        # Checkpoint all 4 epochs as we have not set load_dir to save_dir
        self.mock_trainer_backend, self.mock_module = self._create_mocks()
        self.trainer.args.epochs = 4
        self.trainer = Trainer(
            trainer_backend=self.mock_trainer_backend, module=self.mock_module, args=self.args
        )
        self.trainer.train()
        assert self.trainer.module.on_end_train_epoch.call_count == 4
        assert self._count_files(self.args.checkpointer_args.save_dir, self.args.checkpointer_args.file_ext) == 4
        assert self._count_files(self.args.checkpointer_args.model_state_save_dir, self.args.checkpointer_args.file_ext) == 2

        # Run only 1 additional epoch as we set the load_dir to save_dir
        self.args.checkpointer_args.load_dir = self.args.checkpointer_args.save_dir
        self.mock_trainer_backend, self.mock_module = self._create_mocks()
        self.trainer.args.epochs = 5
        self.trainer = Trainer(
            trainer_backend=self.mock_trainer_backend, module=self.mock_module, args=self.args
        )
        self.trainer.train()
        assert self.trainer.module.on_end_train_epoch.call_count == 1
        assert self._count_files(self.args.checkpointer_args.save_dir, self.args.checkpointer_args.file_ext) == 5
        assert self._count_files(self.args.checkpointer_args.model_state_save_dir, self.args.checkpointer_args.file_ext) == 3
