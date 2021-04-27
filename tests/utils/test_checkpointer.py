"""Module to test default checkpointer class"""
import os
import unittest
from unittest import mock
import torch
from marlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments
from marlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointer
from marlin.utils.checkpointer.checkpoint_utils import Checkpoint

## Wrapper class to allow Mock to be pickable for checkpoint test
class PickableMock(mock.MagicMock):
    def __reduce__(self):
        return (mock.MagicMock, ())


class TestCheckpointer(unittest.TestCase):
    def setUp(self):
        self.args = DefaultCheckpointerArguments(
            checkpoint=False,
            delete_existing_checkpoints=True,
            period=2,
            save_dir=os.path.join("outputs", "save_dir"),
            model_state_save_dir=os.path.join("outputs", "model_save_dir"),
            load_dir=os.path.join("outputs", "load_dir"),
            file_prefix="test",
            file_ext="tar",
        )
        if not os.path.exists(self.args.load_dir):
            os.makedirs(self.args.load_dir)
        self.checkpointer = DefaultCheckpointer(self.args)

    def tearDown(self):
        self._clear_folder(self.args.save_dir, self.args.file_ext)
        self._clear_folder(self.args.model_state_save_dir, self.args.file_ext)
        self._clear_folder(self.args.load_dir, self.args.file_ext)

    def _clear_folder(self, folder, extension):
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(extension):
                    os.remove(os.path.join(folder, f))

    def _count_files(self, folder, extension):
        n_files = 0
        for f in os.listdir(folder):
            if f.endswith(extension):
                n_files += 1
        return n_files

    def test_save(self):
        checkpoint = Checkpoint(module_interface_state=PickableMock(),
                                trainer_state=PickableMock(),
                                trainer_backend_state=PickableMock())

        # Should not checkpoint if checkpoint=False
        self.checkpointer.save(checkpoint, 0)
        assert self._count_files(self.args.save_dir, self.args.file_ext) == 0

        # Should checkpoint if checkpoining is activated,
        # force=True, despite period=2 and index=1
        self.checkpointer.args.checkpoint = True
        self.checkpointer.save(checkpoint, 1, force=True)
        assert self._count_files(self.args.save_dir, self.args.file_ext) == 1

        # Should not checkpoint if force=False and period=2, index=3
        self.checkpointer.save(checkpoint, 3)
        assert self._count_files(self.args.save_dir, self.args.file_ext) == 1

        # Should checkpoint if force=False but period=2 and index=4
        self.checkpointer.save(checkpoint, 4)
        assert self._count_files(self.args.save_dir, self.args.file_ext) == 2

        # Should checkpoint if force=False but period=2 and index=4
        self.checkpointer.save(checkpoint, 4)
        assert self._count_files(self.args.save_dir, self.args.file_ext) == 2

    def test_save_model(self):
        state = PickableMock()
        self.checkpointer.save_model(state, 0)
        assert self._count_files(self.args.model_state_save_dir, self.args.file_ext) == 0
        self.checkpointer.args.checkpoint = True
        self.checkpointer.save_model(state, 0)
        self.checkpointer.save_model(state, 1)
        assert self._count_files(self.args.model_state_save_dir, self.args.file_ext) == 2

    def test_load(self):
        checkpoint = Checkpoint(module_interface_state={},
                                trainer_state={'last_epoch': 0},
                                trainer_backend_state={})

        # Test loading from non-existing folder
        self.checkpointer.args.load_dir = 'non_existing_folder'
        state = self.checkpointer.load()
        assert state == Checkpoint()

        # Verify return None is load_dir is not set
        self.checkpointer.args.load_dir = ''
        state = self.checkpointer.load()
        assert state == Checkpoint()
        self.checkpointer.args.load_dir = None
        state = self.checkpointer.load()
        assert state == Checkpoint()

        # Test loading from empty folder
        self.checkpointer.args.load_dir = os.path.join("outputs", "load_dir")
        state = self.checkpointer.load()
        assert state == Checkpoint()

        # Test loading dummy checkpoint
        path = os.path.join(self.checkpointer.args.load_dir,
            f'{self.checkpointer.args.file_prefix}_{0}.{self.checkpointer.args.file_ext}')
        torch.save(checkpoint.__dict__, path)
        state = self.checkpointer.load()
        assert state.trainer_state['last_epoch'] == 0

        # Test loading specific file
        checkpoint.trainer_state['last_epoch'] = 1
        self.checkpointer.args.load_filename = 'custom_filename.tar'
        torch.save(checkpoint.__dict__, os.path.join(self.checkpointer.args.load_dir,
            self.checkpointer.args.load_filename))
        state = self.checkpointer.load()
        assert state.trainer_state['last_epoch'] == 1

    def test_get_latest_file(self):
        state1 = Checkpoint(module_interface_state=PickableMock(),
                            trainer_state=PickableMock(),
                            trainer_backend_state=PickableMock())
        state2 = Checkpoint(module_interface_state=PickableMock(),
                            trainer_state=PickableMock(),
                            trainer_backend_state=PickableMock())
        path1 = os.path.join(self.checkpointer.args.load_dir,
            f'{self.checkpointer.args.file_prefix}_{1}.{self.checkpointer.args.file_ext}')
        path2 = os.path.join(self.checkpointer.args.load_dir,
            f'{self.checkpointer.args.file_prefix}_{2}.{self.checkpointer.args.file_ext}')
        torch.save(state1.__dict__, path1)
        torch.save(state2.__dict__, path2)
        filepath = self.checkpointer.get_latest_file(self.checkpointer.args.load_dir,
            self.checkpointer.args.file_prefix, self.checkpointer.args. file_ext)
        assert filepath == path2
