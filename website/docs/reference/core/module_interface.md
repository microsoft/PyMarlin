---
sidebar_label: module_interface
title: core.module_interface
---

Module Interface module.

This module contains the abstract classes CallbackInterface and
ModuleInterface that can provide everything necessary for model
training. Users should implement these abstract classes in their
Scenarios.

## Stage Objects

```python
class Stage(enum.Enum)
```

Stage

## CallbackInterface Objects

```python
class CallbackInterface(ABC)
```

A callback class used to add scenario specific outputs/logging/debugging during training.

#### on\_begin\_train\_epoch

```python
 | on_begin_train_epoch(global_step: int, epoch: int)
```

Hook before training epoch (before model forward).

**Arguments**:

- `global_step` _int_ - [description]
- `epoch` _int_ - Current training epoch

#### on\_end\_train\_step

```python
 | on_end_train_step(global_step: int, *train_step_collated_outputs)
```

Runs after end of a global training step.

**Arguments**:

- `global_step` _int_ - current global step
- `train_step_collated_outputs` _list_ - all train step outputs in a list.
  If train_step returns loss, logits train_step_collated_outputs will have [loss_collated, logits_collated]

#### on\_end\_train\_epoch

```python
 | on_end_train_epoch(global_step: int, *train_step_collated_outputs)
```

Hook after training epoch.

**Arguments**:

- `global_step` _int_ - [description]
- `train_step_collated_outputs` _list_ - all train step outputs in a list.
  If train_step returns loss, logits train_step_collated_outputs will have [loss_collated, logits_collated]

#### on\_end\_backward

```python
 | on_end_backward(global_step: int, loss_tensor)
```

Hook after each backward

**Arguments**:

- `global_step` _int_ - [description]
- `loss_tensor(torch.Tensor)` - Undetached loss tensor

#### on\_end\_val\_epoch

```python
 | on_end_val_epoch(global_step: int, *val_step_collated_outputs, *, key="default")
```

Update value at end of end of end of variable

**Arguments**:

- `global_step` _int_ - [description]
  val_step_collated_outputs : all val step outputs in a list.
  If val_step returns loss, logits train_step_collated_outputs will have [loss_collated, logits_collated]
- `key` _str, optional_ - The id of the validation dataloader.
  Defaults to &quot;default&quot;.

#### on\_end\_train

```python
 | on_end_train(global_step: int)
```

Hook after training finishes

**Arguments**:

- `global_step` _int_ - [description]

## ModuleInterface Objects

```python
class ModuleInterface(torch.nn.Module,  CallbackInterface)
```

Interface for PyTorch modules.

This interface contains model architecture in the form of a PyTorch
`nn.Module` together with optimizers and schedules, train and validation
step recipes and any callbacks.

Note: The forward function is overridden.

Note: Users are encouraged to override the `train_step` and `val_step`
methods.

#### get\_optimizers\_schedulers

```python
 | @abstractmethod
 | get_optimizers_schedulers(estimated_global_steps_per_epoch: int, epochs: int) -> Tuple[Iterable[torch.optim.Optimizer], Iterable]
```

Returns a list of optimizers and schedulers
that are used to instantiate the optimizers .

**Returns**:

  Tuple[Iterable[torch.optim.Optimizer], Iterable]:
  list of optimizers and list of schedulers

#### get\_train\_dataloader

```python
 | @abstractmethod
 | get_train_dataloader(sampler: type, batch_size: int) -> torch.utils.data.DataLoader
```

Returns a dataloader for the training loop .
Called every epoch.

**Arguments**:

- `sampler` _type_ - data sampler type which is a derived class of torch.utils.data.Sampler
  Create concrete sampler object before creating dataloader.
- `batch_size` _int_ - batch size per step per device
  

**Returns**:

- `torch.utils.data.DataLoader` - Training dataloader

**Example**:

  train_ds = self.data.get_train_dataset()
  dl = DataLoader(train_ds, batch_size = batch_size, collate_fn= self.collate_fin, sampler = sampler(train_ds))
  return dl

#### get\_val\_dataloaders

```python
 | @abstractmethod
 | get_val_dataloaders(sampler: torch.utils.data.Sampler, batch_size: int) -> Union[
 |         Dict[str, torch.utils.data.DataLoader],
 |         torch.utils.data.DataLoader
 |     ]
```

Returns dataloader(s) for validation loop .
Supports multiple dataloaders based on key value.
Keys will be passed in the callback functions.
Called every epoch .

**Arguments**:

- `sampler` _type_ - data sampler type which is a derived class of torch.utils.data.Sampler
  Create concrete sampler object before creating dataloader.
- `batch_size` _int_ - validation batch size per step per device
  

**Returns**:

  Union[ Dict[str, torch.utils.data.DataLoader],
  torch.utils.data.DataLoader ]:
  A single dataloader or a dictionary of dataloaders
  with key as the data id and value as dataloader

#### get\_test\_dataloaders

```python
 | get_test_dataloaders(sampler, batch_size)
```

Returns test dataloaders

**Arguments**:

- `sampler` _[type]_ - [description]
- `batch_size` _[type]_ - [description]

#### forward

```python
 | forward(stage: Stage, global_step: int, batch, device: Union[torch.device, str, int])
```

torch.nn.Module&#x27;s forward() function.
Overridden to call train_step() or val_step() based on stage .

**Arguments**:

- `stage` _Stage_ - trian/val/test
- `global_step` _int_ - current global step
- `batch` _[type]_ - output of dataloader step
- `device` _Union[torch.device, str, int]_ - device
  

**Raises**:

- `AttributeError` - if stage is different than train, val, test

#### train\_step

```python
 | @abstractmethod
 | train_step(global_step: int, batch, device: Union[torch.device, str, int]) -> Union[torch.Tensor, Tuple]
```

Train a single train step .
Batch should be moved to device before any operation.

**Arguments**:

- `global_step` _int_ - [description]
- `batch` _[type]_ - output of train dataloader step
- `device` _Union[torch.device, str, int]_ - device
  

**Returns**:

  Union[torch.Tensor, Iterable[torch.Tensor]]:
  The first return value must be the loss tensor.
  Can return more than one values in output. All outputs must be tensors
  Callbacks will collate all outputs.

#### val\_step

```python
 | @abstractmethod
 | val_step(global_step: int, batch, device) -> Tuple
```

Runs a single Validation step .

**Arguments**:

- `global_step` _int_ - [description]
- `batch` _[type]_ - [description]
- `device` _[type]_ - [description]

**Returns**:

  Union[torch.Tensor, Iterable[torch.Tensor]]: values that need to be collected - loss, logits etc.
  All outputs must be tensors

#### test\_step

```python
 | test_step(global_step: int, batch, device)
```

Runs a single test step .

**Arguments**:

- `global_step` _int_ - [description]
- `batch` _[type]_ - [description]
- `device` _[type]_ - [description]

#### get\_state

```python
 | get_state()
```

Get the current state of the module, used for checkpointing.

**Returns**:

- `Dict` - Dictionary of variables or objects to checkpoint.

#### update\_state

```python
 | update_state(state: Dict)
```

Update the module from a checkpointed state.

**Arguments**:

- `state` _Dict_ - Output of get_state() during checkpointing.

