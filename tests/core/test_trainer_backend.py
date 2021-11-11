"""Test module for trainer_backend"""

from re import S
import unittest
from unittest import mock
from numpy.testing._private.utils import assert_almost_equal, assert_equal
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
            stage = module_interface.Stage.TRAIN,
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
            stage = module_interface.Stage.VAL,
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

class SimpleArch(torch.nn.Module):
    def __init__(self):
        super(SimpleArch, self).__init__()
        self.net1 = torch.nn.Linear(in_features = 1, out_features = 2, bias = False)
        self.net2 = torch.nn.Linear(2, 1, bias=False)

import numpy as np
class LinearModule(module_interface.ModuleInterface):

    def __init__(self):
        super().__init__()
        self.arch = SimpleArch()
        self.inp = torch.tensor([10,3,5,1,6,9,2,11,35,14]*10).view(-1,1)*1.0
        self.label = self.inp*2
        self.data = list(zip(self.inp,self.label)) # multiplication table of 2
        self.original_weight = self.arch.net1.weight[0][0].item()

    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
        ):
        optimizer = torch.optim.SGD(self.arch.parameters(), lr = 0.01)
        return [optimizer], []

    def get_train_dataloader(
        self, sampler:type, batch_size:int
        ):
        return torch.utils.data.DataLoader(self.data,batch_size = batch_size)
    
    def get_val_dataloaders(
        self, sampler:torch.utils.data.Sampler, batch_size : int
    ):
        return torch.utils.data.DataLoader(self.data,batch_size = batch_size)

    def train_step(
        self, global_step: int, batch, device 
        ):
        inp, label = batch
        print(f"Batch inp", inp)
        pred = self.arch.net2(self.arch.net1(inp))
        print(f"pred {pred}")
        loss = (pred-label).sum()
        return loss
        
    def on_end_backward(self, global_step_completed, loss):
        print(f'global_step_completed = {global_step_completed}')
        # print(f"grad before virtual step= {self.arch.net1.weight.grad}")

    def on_end_train_step(self, global_step_completed, loss):
        print(f'global_step_completed = {global_step_completed}')
        print(f"original weight = {self.original_weight}, current weight = {self.arch.net1.weight[0][0].item()}")

    def val_step(self, global_step: int, batch, device):
        inp, label = batch
        pred = self.net(inp)
        #print(pred, label)


from opacus import PrivacyEngine
import random
# not working
# Number batches seen = max_train_steps_per epoch * gradient_accumulation ( 4*2 == 8*1)
torch.manual_seed(40)
random.seed(40)
np.random.seed(40)
import copy
class TestSingleProcessDpSgdWithSingleWeight(unittest.TestCase):
    
    def setUp(self):
        
        self.trainer_backend = trainer_backend.SingleProcessDpSgd()
        torch.manual_seed(40)
        self.model = LinearModule()
        self.trainer_backendArgs = trainer_backend.TrainerBackendArguments(
            model = self.model,
            device = 'cpu',
            max_train_steps_per_epoch= 1,
            max_val_steps_per_epoch = 1,
            distributed_training_args = DistributedTrainingArguments(),
            optimizers = self.model.get_optimizers_schedulers(1,1)[0],
            schedulers = [],
            gradient_accumulation=1,
            clip_grads=False,
            opacus_params={'max_grad_norm': 1.0, 'sample_rate': 0.01, 'noise_multiplier':1.3}
        )
        self.trainer_backend.init(self.trainer_backendArgs)
        self.trainer_backend.privacy_engine._set_seed(40)
    
    #### DP Model without using PyMarlin trainer ####
    def setUp_PyTorch_dpVirtual(self):
        torch.manual_seed(40)
        self.pt_model = LinearModule()
        print("----------- Initiating first model with virtual step")
        self.pt_optimizer = self.pt_model.get_optimizers_schedulers(1,1)[0][0] 
        self.pt_privacy_engine = PrivacyEngine(self.pt_model, sample_rate=0.01,
                                        noise_multiplier=1.3, max_grad_norm=1.0)
        self.pt_privacy_engine.attach(self.pt_optimizer)
        self.pt_privacy_engine._set_seed(40)

    def setUp_dpVirtual(self):
        torch.manual_seed(40)
        self.model2=LinearModule()
        print("----------- Initiating second model with virtual step")
        self.trainer_backend = trainer_backend.SingleProcessDpSgd()
        self.trainer_backendArgs = trainer_backend.TrainerBackendArguments(
            model = self.model2,
            device = 'cpu',
            max_train_steps_per_epoch= 1,
            max_val_steps_per_epoch = 1,
            distributed_training_args = DistributedTrainingArguments(),
            optimizers = self.model2.get_optimizers_schedulers(1,1)[0],
            schedulers = [],
            gradient_accumulation=2,# To force virtual steps
            clip_grads=False,
            opacus_params={'max_grad_norm': 1.0, 'sample_rate': 0.01, 'noise_multiplier':1.3},
        )
        self.trainer_backend.init(self.trainer_backendArgs)
        self.trainer_backend.privacy_engine._set_seed(40)
        
    def setUp_vanillaSP(self):
        torch.manual_seed(40)
        self.model_simple = LinearModule()
        print("----------- Initiating third model with vanilla SP")
        self.trainer_backend = trainer_backend.SingleProcess()
        self.trainer_backendArgs = trainer_backend.TrainerBackendArguments(
            model = self.model_simple,
            device = 'cpu',
            max_train_steps_per_epoch= 1,
            max_val_steps_per_epoch = 1,
            distributed_training_args = DistributedTrainingArguments(),
            optimizers = self.model_simple.get_optimizers_schedulers(1,1)[0],
            schedulers = [],
            gradient_accumulation=1,
            clip_grads=False,
        )
        self.trainer_backend.init(self.trainer_backendArgs)
    
    def setUp_vanillaSP_virtual(self):
        torch.manual_seed(40)
        self.model_simple_virtual = LinearModule()
        print("----------- Initiating fourth model with vanilla SP + virtual")
        self.trainer_backend = trainer_backend.SingleProcess()
        self.trainer_backendArgs = trainer_backend.TrainerBackendArguments(
            model = self.model_simple_virtual,
            device = 'cpu',
            max_train_steps_per_epoch= 1,
            max_val_steps_per_epoch = 1,
            distributed_training_args = DistributedTrainingArguments(),
            optimizers = self.model_simple_virtual.get_optimizers_schedulers(1,1)[0],
            schedulers = [],
            gradient_accumulation=2,
            clip_grads=False,
        )
        self.trainer_backend.init(self.trainer_backendArgs)

    def test_train_step_PyTorch_PyMarlin_dpVirtual(self):

        pt_virtual_steps = 2
        pt_virtual_batch_size = 4 // pt_virtual_steps

        # PyTorch DP model with virtual steps
        self.setUp_PyTorch_dpVirtual()
        for k in range(pt_virtual_steps):
            virtual_batch = slice(k * pt_virtual_batch_size, (k + 1) * pt_virtual_batch_size)
            x, y = self.pt_model.inp[virtual_batch], self.pt_model.label[virtual_batch]
            loss = self.pt_model.train_step(1, (x,y), None)
            print(f"before loss/ga = {loss}")
            loss = loss / pt_virtual_steps
            print(f"after loss/ga = {loss}")
            loss.backward()
            if k < pt_virtual_steps - 1:
                print(f"grad before virtual step {self.pt_model.arch.net1.weight.grad[0].item()}")
                self.pt_optimizer.virtual_step()
                print(f"wt after virtual step {self.pt_model.arch.net1.weight[0][0]}")
            else:
                print(f"grad before regular step {self.pt_model.arch.net1.weight.grad[0].item()}")
                self.pt_optimizer.step()
                self.pt_optimizer.zero_grad()
                print(f"wt after regular step {self.pt_model.arch.net1.weight[0][0]}")
        diff_delta1 = self.pt_model.arch.net1.weight[0][0] - self.pt_model.original_weight
        print("The delta in weight after PyTorch Dp training WITH virtual: ", diff_delta1)
        print("Original model weight: ", self.pt_model.original_weight)
        print("modified model weight: ", self.pt_model.arch.net1.weight[0][0].item())

        # PyMarlin DP model with virtual steps
        self.setUp_dpVirtual()
        self.trainer_backend.train_dl(self.model2.get_train_dataloader(sampler = None, batch_size = pt_virtual_batch_size), self.model2)
        diff_delta2 = self.model2.arch.net1.weight[0][0] - self.model2.original_weight
        print("The delta in weight after PyMarlin Dp training WITH virtual: ", diff_delta2)
        print("Original model weight: ", self.model2.original_weight)
        print("modified model weight: ", self.model2.arch.net1.weight[0][0].item())

        print(f"Diff delta PyTorch DP = {diff_delta1}")
        print(f"Diff delta PyMarlin DP = {diff_delta2}")

        ## GA = 2
        assert diff_delta1.item() == 0.0017529129981994629
        assert diff_delta2.item() == 0.0018999576568603516

    def test_train_dl_novirtual(self):
        # DP model with no virtual steps
        self.trainer_backend.train_dl(self.model.get_train_dataloader(sampler = None, batch_size = 4), self.model)
        diff_delta = self.model.arch.net1.weight[0][0] - self.model.original_weight
        print("The delta in weight after PyMarlin Dp training with no virtual: ", diff_delta)
        print("Original model weight: ", self.model.original_weight)
        print("modified model weight: ", self.model.arch.net1.weight[0][0].item())
        assert diff_delta.item() == 0.001954376697540283 ## after 1 step with no virtual
        self.no_virtual_weight = self.model.arch.net1.weight[0][0].item()
        
        # DP model with virtual steps
        self.setUp_dpVirtual()
        self.trainer_backend.train_dl(self.model2.get_train_dataloader(sampler = None, batch_size = 2), self.model2)
        diff_delta = self.model2.arch.net1.weight[0][0] - self.model.original_weight
        print("The delta in weight after PyMarlin Dp training WITH virtual: ", diff_delta)
        print("Original model weight: ", self.model.original_weight)
        print("modified model weight: ", self.model2.arch.net1.weight[0][0].item())
        assert diff_delta.item() == 0.0018999576568603516
        self.virtual_weight = self.model2.arch.net1.weight[0][0].item()
        
        # Regular SP training
        self.setUp_vanillaSP()
        self.trainer_backend.train_dl(self.model_simple.get_train_dataloader(sampler = None, batch_size = 4), self.model_simple)
        diff_delta = self.model_simple.arch.net1.weight[0][0] - self.model.original_weight
        print("The delta in weight after SP training: ", diff_delta)
        print("Original model weight: ", self.model.original_weight)
        print("modified model weight: ", self.model_simple.arch.net1.weight[0][0].item())
        assert diff_delta.item() == 0.08767524361610413
        self.sp_weight = self.model_simple.arch.net1.weight[0][0].item()

        # Regular SP with virtual steps
        self.setUp_vanillaSP_virtual()
        self.trainer_backend.train_dl(self.model_simple_virtual.get_train_dataloader(sampler = None, batch_size = 2), self.model_simple_virtual)
        diff_delta = self.model_simple_virtual.arch.net1.weight[0][0] - self.model.original_weight
        print("The delta in weight after SP training with virtual: ", diff_delta)
        print("Original model weight: ", self.model.original_weight)
        print("modified model weight: ", self.model_simple_virtual.arch.net1.weight[0][0].item())
        assert diff_delta.item() == 0.04383762180805206
        self.sp_virtual_weight = self.model_simple_virtual.arch.net1.weight[0][0].item()

        Sp_virtual_diff = (self.sp_weight - self.sp_virtual_weight)
        assert Sp_virtual_diff == 0.04383762180805206 # asserts delta with and without virtual for SP

        DP_Training_weight_diff = (self.no_virtual_weight- self.virtual_weight)
        assert DP_Training_weight_diff == 5.441904067993164e-05 # asserts delta with and without virtual step, both DP

        Vanilla_training_diff = (self.sp_weight - self.no_virtual_weight)
        assert Vanilla_training_diff == 0.08572086691856384 # asserts delta with and without DP ( no virtual)
