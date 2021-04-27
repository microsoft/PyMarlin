# CIFAR image classification

Krishan Subudhi 01/27/2021

This tutorial is based on official PyTorch blog on [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) which trains a image classifier using CIFAR data.



```python
#!pip install torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```

## Step 1. Data preprocessing

This step involves, 
1. Downloading data
2. Preprocessing it 
3. Analyzing it
4. Creating a final dataset

In marlin, **DataInteface** and **DataProcessor** is where you implement the code related to all the steps above.


```python
from marlin.core import data_interface
```


```python
class CifarDataProcessor(data_interface.DataProcessor):
    def process(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        return {'Train': trainset, 'Test': testset}

    def analyze(self, datasets):
        print(f'train data size = {len(datasets["Train"])}')
        print(f'val data size = {len(datasets["Test"])}')
        print('Examples')
        sample_images = [datasets['Train'][i][0] for i in range(4)]
        self._imshow(torchvision.utils.make_grid(sample_images))
        
    def _imshow(self,img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

```


```python
class CifarDataInterface(data_interface.DataInterface):
    
    def setup_datasets(self, train_ds, val_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds
        
    @property
    def classes(self):
        return ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def get_train_dataset(self):
        return self.train_ds
    
    def get_val_dataset(self):
        return self.val_ds 
```


```python
dp = CifarDataProcessor()
dm = CifarDataInterface()
datasets = dm.process_data(dp)
```

    Files already downloaded and verified
    Files already downloaded and verified
    train data size = 50000
    val data size = 10000
    Examples
    


    
![png](images/cifar_1.png)
    


## Step 2. Training


```python
from marlin.core import module_interface
```


```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```


```python
class CifarModule(module_interface.ModuleInterface):
    '''
    ModuleInterface contains instruction to create data loader , 
    defines train step, optimizer, scheduler, evaluation etc.
    
    Just implement the abstract function: refer docstrings.
    '''
    def __init__(self, data_interface):
        super().__init__() # always initialize superclass first
        self.data_interface = data_interface
        
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        
        self.running_loss = 0.0

    def get_optimizers_schedulers(
        self, estimated_global_steps_per_epoch: int, epochs: int
        ):
        return [self.optimizer], []

    def get_train_dataloader(
        self, sampler:type, batch_size:int
        ):
        print('Inside get_train_dataloader',batch_size)
        return torch.utils.data.DataLoader(self.data_interface.get_train_dataset(), batch_size=batch_size,
                                          shuffle=True)

    def get_val_dataloaders(
        self, sampler:torch.utils.data.Sampler, batch_size : int
        ): 
        return torch.utils.data.DataLoader(self.data_interface.get_val_dataset(), batch_size=batch_size,
                                         shuffle=False)

    def train_step(
        self, global_step: int, batch, device
        ):
        '''
        First output should be loss. Can return multiple outputs
        '''
        inputs, labels = batch # output of dataloader will be input of train_step
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        self.running_loss += loss.item()
        if global_step % 2000 == 0:    # print every 2000 mini-batches
            print('[%5d] loss: %.3f' %
                  (global_step, self.running_loss / 2000))
            self.running_loss = 0.0
        return loss

    def val_step(self, global_step: int, batch, device) :
        '''
        Can return multiple outputs. First output need not be loss.
        '''
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = self.net(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct, total

    def on_end_val_epoch(self,
        global_step: int,
        *val_step_collated_outputs,
        key='default'):
        '''
        callback after validation loop ends
        '''
        corrects, totals = val_step_collated_outputs
        correct = sum(corrects) # list of integers
        total= sum(totals)
        
        accuracy = 100 * correct / total
        print(f'Val accuracy at step {global_step} = {accuracy}%')
        
```


```python
dm.setup_datasets(datasets['Train'], datasets['Test'])
module = CifarModule(dm)
```

### Train for few steps
Check if the entire loop runs without error. Use `max_train_steps_per_epoch` and `max_val_steps_per_epoch` to stop early. Set them to `null` to train on full data.

```python
from marlin.core import trainer, trainer_backend
from marlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments
backend = trainer_backend.SingleProcess()
chkp_args = DefaultCheckpointerArguments(checkpoint=False)

args = trainer.TrainerArguments(
    epochs=2,
    max_train_steps_per_epoch = 100,
    max_val_steps_per_epoch = 10,
    train_batch_size=4,
    val_batch_size=16,
    writers=[],
    log_level = 'DEBUG'
)
tr = trainer.Trainer(
    trainer_backend = backend,
    module = module,
    args = args
)
```

    Inside get_train_dataloader 4
    Inside get_train_dataloader 4
    SystemLog: 2021-04-01 21:39:57,419:INFO : marlin.core.trainer : 219 : _abc_impl: <_abc_data object at 0x000002489C5B1B70>
    SystemLog: 2021-04-01 21:39:57,420:INFO : marlin.core.trainer : 219 : args: TrainerArguments(epochs=2, use_gpu=True, train_batch_size=4, gpu_batch_size_limit=512, val_batch_size=16, max_train_steps_per_epoch=100, max_val_steps_per_epoch=10, clip_grads=True, max_grad_norm=1.0, reset_optimizers_schedulers=False, checkpointer_args=DefaultCheckpointerArguments(checkpoint=True, delete_existing_checkpoints=False, period=1, save_dir='your\\path', model_state_save_dir='your\\path', load_dir=None, load_filename=None, file_prefix='model', file_ext='pt', log_level='INFO'), distributed_training_args=DistributedTrainingArguments(local_rank=0, global_rank=0, world_size=1, backend='nccl', init_method='env://', gather_frequency=None), writers=[], stats_args=StatInitArguments(log_steps=1, update_system_stats=False, log_model_steps=1000, exclude_list='bias|LayerNorm|layer\\.[3-9]|layer\\.1(?!1)|layer\\.2(?!3)'), writer_args=WriterInitArguments(tb_log_dir='logs', tb_logpath_parent_env=None, tb_log_multi=False, tb_log_hist_steps=20000, model_log_level='INFO'), disable_tqdm=False, log_level='DEBUG', backend='sp')
    SystemLog: 2021-04-01 21:39:57,421:INFO : marlin.core.trainer : 219 : device: cpu
    SystemLog: 2021-04-01 21:39:57,421:INFO : marlin.core.trainer : 219 : estimated_global_steps_per_epoch: 100
    SystemLog: 2021-04-01 21:39:57,421:INFO : marlin.core.trainer : 219 : global_steps_finished: 0
    SystemLog: 2021-04-01 21:39:57,422:INFO : marlin.core.trainer : 219 : gradient_accumulation: 1
    SystemLog: 2021-04-01 21:39:57,422:INFO : marlin.core.trainer : 219 : is_distributed: False
    SystemLog: 2021-04-01 21:39:57,423:INFO : marlin.core.trainer : 219 : is_main_process: True
    SystemLog: 2021-04-01 21:39:57,424:INFO : marlin.core.trainer : 219 : logger: <Logger marlin.core.trainer (DEBUG)>
    SystemLog: 2021-04-01 21:39:57,424:INFO : marlin.core.trainer : 219 : pergpu_global_batch_size: 4
    SystemLog: 2021-04-01 21:39:57,424:INFO : marlin.core.trainer : 219 : stats: <marlin.utils.stats.basic_stats.BasicStats object at 0x000002489C594610>
    SystemLog: 2021-04-01 21:39:57,425:INFO : marlin.core.trainer : 219 : total_steps_finished: 0
    SystemLog: 2021-04-01 21:39:57,425:INFO : marlin.core.trainer : 219 : train_step_batch_size: 4
    SystemLog: 2021-04-01 21:39:57,425:INFO : marlin.core.trainer : 219 : trainer_backend: <marlin.core.trainer_backend.SingleProcess object at 0x000002489A69F3A0>
    SystemLog: 2021-04-01 21:39:57,426:INFO : marlin.core.trainer : 219 : val_step_batch_size: 16
    


```python
tr.train()
```
      0%|          | 0/2 [00:00<?, ?it/s]

    SystemLog: 2021-04-01 21:40:03,871:INFO : marlin.core.trainer : 141 : Training epoch 0

      0%|          | 0/12500 [00:00<?, ?batch/s]

    SystemLog: 2021-04-01 21:40:04,457:INFO : marlin.core.trainer : 147 : Validating  
    
      0%|          | 0/625 [00:00<?, ?it/s]

    Val accuracy at step 100 = 15.625%
    SystemLog: 2021-04-01 21:40:04,515:INFO : marlin.core.trainer : 141 : Training epoch 1    

      0%|          | 0/12500 [00:00<?, ?batch/s]

    SystemLog: 2021-04-01 21:40:05,096:INFO : marlin.core.trainer : 147 : Validating 

      0%|          | 0/625 [00:00<?, ?it/s]

    Val accuracy at step 200 = 10.0%
    SystemLog: 2021-04-01 21:40:05,160:INFO : marlin.core.trainer : 161 : Finished training ..
    

### Final Training


```python
from marlin.core import trainer, trainer_backend
from marlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments
backend = trainer_backend.SingleProcess()
chkp_args = DefaultCheckpointerArguments(checkpoint=False)

args = trainer.TrainerArguments(
    epochs=2,
    train_batch_size=4,
    val_batch_size=16,
    writers=['tensorboard'],
    clip_grads=False,
    log_level = 'INFO',
    checkpointer_args=chkp_args
)

tr = trainer.Trainer(
    trainer_backend = backend,
    module = module,
    args = args
)

tr.train()
```

    Inside get_train_dataloader 4
    Inside get_train_dataloader 4
    SystemLog: 2021-04-01 21:40:25,953:INFO : marlin.core.trainer : 219 : _abc_impl: <_abc_data object at 0x000002489C5B1B70>
    SystemLog: 2021-04-01 21:40:25,953:INFO : marlin.core.trainer : 219 : args: TrainerArguments(epochs=2, use_gpu=True, train_batch_size=4, gpu_batch_size_limit=512, val_batch_size=16, max_train_steps_per_epoch=None, max_val_steps_per_epoch=None, clip_grads=False, max_grad_norm=1.0, reset_optimizers_schedulers=False, checkpointer_args=DefaultCheckpointerArguments(checkpoint=False, delete_existing_checkpoints=False, period=1, save_dir='your\\path', model_state_save_dir='your\\path', load_dir=None, load_filename=None, file_prefix='model', 
    file_ext='pt', log_level='INFO'), distributed_training_args=DistributedTrainingArguments(local_rank=0, global_rank=0, world_size=1, backend='nccl', init_method='env://', gather_frequency=None), writers=['tensorboard'], stats_args=StatInitArguments(log_steps=1, update_system_stats=False, log_model_steps=1000, exclude_list='bias|LayerNorm|layer\\.[3-9]|layer\\.1(?!1)|layer\\.2(?!3)'), writer_args=WriterInitArguments(tb_log_dir='logs', tb_logpath_parent_env=None, tb_log_multi=False, tb_log_hist_steps=20000, model_log_level='INFO'), disable_tqdm=False, log_level='INFO', backend='sp')
    SystemLog: 2021-04-01 21:40:25,953:INFO : marlin.core.trainer : 219 : device: cpu
    SystemLog: 2021-04-01 21:40:25,954:INFO : marlin.core.trainer : 219 : estimated_global_steps_per_epoch: 12501
    SystemLog: 2021-04-01 21:40:25,954:INFO : marlin.core.trainer : 219 : global_steps_finished: 0
    SystemLog: 2021-04-01 21:40:25,954:INFO : marlin.core.trainer : 219 : gradient_accumulation: 1
    SystemLog: 2021-04-01 21:40:25,954:INFO : marlin.core.trainer : 219 : is_distributed: False
    SystemLog: 2021-04-01 21:40:25,955:INFO : marlin.core.trainer : 219 : is_main_process: True
    SystemLog: 2021-04-01 21:40:25,955:INFO : marlin.core.trainer : 219 : logger: <Logger marlin.core.trainer (INFO)>
    SystemLog: 2021-04-01 21:40:25,955:INFO : marlin.core.trainer : 219 : pergpu_global_batch_size: 4
    SystemLog: 2021-04-01 21:40:25,956:INFO : marlin.core.trainer : 219 : stats: <marlin.utils.stats.basic_stats.BasicStats object at 0x000002489C594610>
    SystemLog: 2021-04-01 21:40:25,957:INFO : marlin.core.trainer : 219 : total_steps_finished: 0
    SystemLog: 2021-04-01 21:40:25,957:INFO : marlin.core.trainer : 219 : train_step_batch_size: 4
    SystemLog: 2021-04-01 21:40:25,958:INFO : marlin.core.trainer : 219 : trainer_backend: <marlin.core.trainer_backend.SingleProcess object at 0x000002489CBD45E0>
    SystemLog: 2021-04-01 21:40:25,958:INFO : marlin.core.trainer : 219 : val_step_batch_size: 16
    Inside get_train_dataloader 4
    SystemLog: 2021-04-01 21:40:25,960:INFO : marlin.utils.writer.tensorboard : 43 : Cleared directory logs (skipping azureml dirs)
    SystemLog: 2021-04-01 21:40:25,961:INFO : marlin.utils.writer.tensorboard : 46 : Created tensorboard folder logs : []
    
      0%|          | 0/2 [00:00<?, ?it/s]

      0%|          | 0/12500 [00:00<?, ?batch/s]

    [ 2000] loss: 2.395
    [ 4000] loss: 1.823
    [ 6000] loss: 1.653
    [ 8000] loss: 1.568
    [10000] loss: 1.495
    [12000] loss: 1.430
    
      0%|          | 0/625 [00:00<?, ?it/s]

    Val accuracy at step 12500 = 49.7%

      0%|          | 0/12500 [00:00<?, ?batch/s]

    [14000] loss: 1.406
    [16000] loss: 1.358
    [18000] loss: 1.324
    [20000] loss: 1.301
    [22000] loss: 1.288
    [24000] loss: 1.269
    
      0%|          | 0/625 [00:00<?, ?it/s]

    Val accuracy at step 25000 = 55.35%
    SystemLog: 2021-04-01 21:43:59,683:INFO : marlin.core.trainer : 161 : Finished training ..
    

## Step 3. Saving the model
```python
PATH = './cifar_net.pth'
torch.save(module.net.state_dict(), PATH)
```
