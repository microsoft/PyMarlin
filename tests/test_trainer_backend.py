"""Test module for trainer_backend"""

import unittest
from unittest import mock
import torch
import pytest
from pymarlin.core import module_interface, trainer_backend
from pymarlin.utils.distributed import DistributedTrainingArguments
#https://docs.python.org/3/library/unittest.mock.hmock_tml

class TestSingleProcess(unittest.TestCase):
    def setUp(self):
        self.trainer_backend = trainer_backend.SingleProcess()
        self.mock_module = mock.MagicMock(spec = module_interface.ModuleInterface)
        # make x^2 as loss
        # self.x = torch.Tensor([1])
        # self.x.requires_grad = True
        # self.loss = self.x*self.x
        self.loss = torch.randn(1, requires_grad=True)
        self.mock_module.forward = mock.MagicMock(return_value = [self.loss])

        self.mock_scheduler = mock.MagicMock()

        self.mock_optimizer = mock.MagicMock(spec = torch.optim.Optimizer)
        self.trainer_backendArgs = trainer_backend.TrainerBackendArguments(
            model = self.mock_module,
            device = 'cpu',
            max_train_steps_per_epoch= 1,
            max_val_steps_per_epoch = 1,
            distributed_training_args = DistributedTrainingArguments(),
            optimizers = [self.mock_optimizer],
            schedulers = [self.mock_scheduler],
            gradient_accumulation=1,
            clip_grads=False,
        )
        
        self.trainer_backend.init(self.trainer_backendArgs)

        
        self.mock_callback = mock.MagicMock()
        self.mock_dataloader = [mock.MagicMock()]*10

    def test_train_dl(self):

        # make x^2 as loss
        x = torch.Tensor([1])
        x.requires_grad = True
        loss = x*x
        self.mock_module.forward = mock.MagicMock(return_value = [loss])

        
        self.trainer_backend.train_dl(self.mock_dataloader, self.mock_callback)


        # test forward
        self.mock_module.forward.assert_called_once_with(
            stage = module_interface.Stage.train,
            batch = self.mock_dataloader[0],
            device = 'cpu',
            global_step =1 )
        print(self.mock_module.forward.return_value)
        # test backward
        assert x.grad == 2 *x
        # test optimization
        self.mock_optimizer.step.assert_called_once()
        self.mock_optimizer.zero_grad.assert_called_once()
        #test callback
        self.mock_callback.on_end_train_step.assert_called_once()
        self.mock_callback.on_end_train_step.assert_called_with( 1, loss.detach())

    def test_eval_dl(self):

        self.trainer_backend.validate_dl(self.mock_dataloader)

        # test forward
        self.mock_module.forward.assert_called_once_with(
            stage = module_interface.Stage.val,
            batch = self.mock_dataloader[0],
            device = 'cpu',
            global_step = 0 )

    def test_gradiend_accumulation(self):
        self.trainer_backend.args.gradient_accumulation = 2
        self.trainer_backend.train_dl(self.mock_dataloader, self.mock_callback)
        assert self.mock_module.forward.call_count == 2
        assert self.mock_optimizer.step.call_count == 1
        assert self.mock_optimizer.step.call_count == 1

    def test_gradiend_clipping(self):
        # make x^2 as loss
        

        self.trainer_backend.args.clip_grads = True
        self.trainer_backend.args.max_grad_norm  = 1

        for val in range(-10, 10):
            x = torch.Tensor([val])
            x.requires_grad = True
            loss = x*x
            self.mock_module.parameters = mock.MagicMock(return_value = [x])
            self.mock_module.forward = mock.MagicMock(return_value = [loss])
            self.trainer_backend.train_dl(self.mock_dataloader, self.mock_callback)

            assert min(0, 2*val -1) < x.grad.item() <= 1

    def test_output_collection(self):

        self.trainer_backendArgs.max_train_steps_per_epoch = 2
        self.trainer_backend.args.gradient_accumulation = 2

        losses = [torch.randn(1, requires_grad=True).squeeze(), torch.randn(1, requires_grad=True).squeeze()] * 2
        labels = [torch.randint(0,10, size = (4,3)), torch.randint(0,10, size = (3,3))] * 2
        # guids = range(4)
        self.mock_module.forward = mock.MagicMock()
        self.mock_module.forward.side_effect = zip(losses, labels)#, guids)

        outputs = self.trainer_backend.train_dl(self.mock_dataloader, self.mock_callback)
        assert self.mock_module.forward.call_count == 4

        assert outputs[0].shape == torch.Size([4])
        assert outputs[1].shape == torch.Size([4+3+4+3, 3]) # concatinated across axis 0
        # assert outputs[2] == [0,1,2,3] 
        #print(outputs, outputs[0].shape)


    def test_get_state(self):
        state = self.trainer_backend.get_state()
        assert state['global_step_completed'] == 0
        assert state['batches_completed'] == 0

    def test_update_state(self):
        state_dict = {
            'global_step_completed': 1,
            'batches_completed': 2
        }
        self.trainer_backend.update_state(state_dict)
        assert self.trainer_backend.get_global_steps_completed() == 1
        assert self.trainer_backend.get_batches_completed() == 2


@pytest.mark.filterwarnings("ignore::UserWarning: torch.cuda.amp.")
class TestSingleProcessAmp(TestSingleProcess):
    def setUp(self):
        super().setUp()
        self.trainer_backend = trainer_backend.SingleProcessAmp()
        self.trainer_backend.init(self.trainer_backendArgs)