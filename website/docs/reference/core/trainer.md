---
sidebar_label: trainer
title: core.trainer
---

Trainer module.

## TrainerArguments Objects

```python
@dataclass
class TrainerArguments()
```

Trainer Arguments class.

## AbstractTrainer Objects

```python
class AbstractTrainer(ABC)
```

Abstract Trainer class.

#### train

```python
 | @abstractmethod
 | train()
```

Run Train loop

#### validate

```python
 | @abstractmethod
 | validate()
```

Run eval loop

## Trainer Objects

```python
class Trainer(AbstractTrainer)
```

Orchestrates model training.

The `Trainer` is responsible for coordinating the model definition
(`ModuleInterface`) and the `TrainerBackend` - connecting the high-level
model recipe with the backend on which it will be trained.

This accepts a `module` implementing `ModuleInterface` - it contains the
model definition, as well as the definition of train and evaluation steps,
optimizers and schedulers and any optional callbacks.

It also accepts a `TrainerBackend` defining how the training should be run
e.g. single node vs distributed training. There are `TrainerBackends` for
most common scenarios available out of the box - or alternatively a user can
provide a custom `TrainerBackend`.

**Arguments**:

- `module` _ModuleInterface_ - Contains model definition, train and validation
  definition, optimizer and scheduler, and optional callbacks.
- `args` _TrainerArguments_ - Training hyperparameters.
  
  Optional keyword arguments:
- `trainer_backend` _TrainerBackend_ - How the training will be carried out.
  For example, the training is distributed and/or using AMP (automatic mixed precision).
  This can also be specified in args using the backend keyword.
  Defaults to singleprocess. Options are: sp (singleprocess), sp-amp, ddp, ddp-amp.
- `checkpointer` _AbstractCheckpointer_ - Used to handle model checkpointing.

#### \_\_init\_\_

```python
 | __init__(module: module_interface.ModuleInterface, args: TrainerArguments, trainer_backend: Optional[trn.TrainerBackend] = None, checkpointer: Optional[AbstractCheckpointer] = None)
```

Initializes stats, writers, trainer_backend and other helper functions

#### train

```python
 | train()
```

Train and validate the model

#### validate

```python
 | validate()
```

Run evaluation over multiple validation dataloaders

#### save\_checkpoint

```python
 | save_checkpoint(force=False) -> None
```

Checkpoint the current state of the Trainer, TrainerBackend, and ModuleInterface.

Saves state of each object in a dictionary by calling on their get_state() methods and
providing the states to the checkpointer&#x27;s save() method.

#### save\_model\_checkpoint

```python
 | save_model_checkpoint() -> None
```

Checkpoint the current state of the ModuleInterface, used to save the final model in the
training loop.

Saves state of the ModuleInterface by calling on it&#x27;s get_state() method and providing it
to the checkpointer&#x27;s save_model() method.

#### load\_checkpoints

```python
 | load_checkpoints() -> Checkpoint
```

Load state of Trainer, TrainerBackend, and ModuleInterface from checkpoint.

Loading logic is determined by the checkpointer used, see DefaultCheckpointer
for default implementation logic. If a checkpoint is loaded, all module
states are updated.

#### get\_state

```python
 | get_state() -> dict
```

Get the current state of the Trainer for checkpointing.

Default implementation returns epochs finished, override to include
aditional state properties.

**Returns**:

- `state_dict` _dict_ - Dictionary of variables or objects to checkpoint.

#### update\_state

```python
 | update_state(state: dict) -> None
```

Update the Trainer&#x27;s state from a checkpointed state.

**Arguments**:

  state : Output of get_state() during checkpointing.

#### device

```python
 | @property
 | device()
```

The torch device either CPU or GPU, and the local rank.

Note: _fetch_rank() should have already been called
before calling device.

#### train\_step\_batch\_size

```python
 | @property
 | train_step_batch_size()
```

Returns micro batch sizes for training. Splits batch into smaller step batches such that
    1. Each step batch fits into memory
    2. step batch size are a factor of global batch size per gpu
    3. attain maxium batch size that follows 1 and 2

#### estimated\_global\_steps\_per\_epoch

```python
 | @property
 | estimated_global_steps_per_epoch()
```

Estimate the number of global steps per epoch.

Compute the maximum number of global steps as len(dataloader) // gradient_accumulation + 1.
If max_train_steps_per_epoch is provided we return the minimum of the two.

Note: SequentialSampler is used to get the train dataloader regardless of
the sampler provided by trainer_backend as we only require the length of the dataloader.

Do not change this logic without testing thorougly. There is a test case already written.

TODO: simplify this by initiliaizing distributed environment before calling this and remove SequentialSampler.

