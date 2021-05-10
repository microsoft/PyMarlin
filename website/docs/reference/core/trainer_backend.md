---
sidebar_label: trainer_backend
title: core.trainer_backend
---

module for all trainer_backend classes.

#### build\_trainer\_backend

```python
build_trainer_backend(trainer_backend_name, *args, **kwargs)
```

Factory for trainer_backends

**Arguments**:

- `trainer_backend_name` _str_ - TrainerBackend Name. Possible choices are currently: sp, sp-amp, ddp, ddp-amp
- `args` _sequence_ - TrainerBackend positional arguments
- `kwargs` _dict_ - TrainerBackend keyword arguments

## OutputCollector Objects

```python
class OutputCollector()
```

Responsible for collecting step outputs and stores them in memory across each call.
Concatinates tensors from all steps across first dimension.

#### collect

```python
 | collect(outputs: Union[torch.Tensor, Iterable[torch.Tensor]])
```

Coalesces train_step and val_step outputs.
all tensors concatenated across dimension 0
if input is a torch.Tensor of dimension batch_size * x* y .., all_outputs will be List[torch.Tensor of dimension total_samples_till_now *x *y]
if input is a torch.Tensor of dimension 1 * 1, all_outputs will List[torch.Tensor of dimension total_samples_till_now * 1]
if input is List[torch.Tensor], all_outputs will be List[torch.Tensor] - all tensors concatenated across dimension 0

**Arguments**:

- `outputs` _Union[torch.Tensor, Iterable[torch.Tensor]]_ - train_step , val_step outputs

## SingleProcess Objects

```python
class SingleProcess(TrainerBackend)
```

Single Process TrainerBackend

#### \_\_init\_\_

```python
 | __init__()
```

Single process trainer_backend

#### process\_global\_step

```python
 | process_global_step(global_step_collector, callback)
```

Clip gradients and call optimizer + scheduler

#### get\_state

```python
 | get_state() -> dict
```

Get the current state of the trainer_backend, used for checkpointing.

**Returns**:

- `state_dict` _dict_ - Dictionary of variables or objects to checkpoint.

#### update\_state

```python
 | update_state(state) -> None
```

Update the trainer_backend from a checkpointed state.

**Arguments**:

  state (dict) : Output of get_state() during checkpointing

## AbstractTrainerBackendDecorator Objects

```python
class AbstractTrainerBackendDecorator(TrainerBackend)
```

Abstract class implementing the decorator design pattern.

## DDPTrainerBackend Objects

```python
class DDPTrainerBackend(AbstractTrainerBackendDecorator)
```

Distributed Data Parallel TrainerBackend.

Wraps ModuleInterface model with DistributedDataParallel which handles
gradient averaging across processes.

.. note: Assumes initiailized model parameters are consistent across
    processes - e.g. by using same random seed in each process at
    point of model initialization.

#### setup\_distributed\_env

```python
 | setup_distributed_env()
```

Setup the process group for distributed training.

#### cleanup

```python
 | cleanup()
```

Destroy the process group used for distributed training.

#### gather\_tensors\_on\_cpu

```python
 | gather_tensors_on_cpu(x: torch.tensor)
```

Gather tensors and move to cpu at configurable frequency.

Move tensor to CUDA device, apply all-gather and move back to CPU.
If `distributed_training_args.gather_frequency` is set,  tensors are
moved to CUDA in chunks of that size.

**Arguments**:

- `x` _torch.tensor_ - To be gathered.
  

**Returns**:

  Gathered tensor on the cpu.

