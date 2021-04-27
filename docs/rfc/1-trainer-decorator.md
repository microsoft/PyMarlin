# Trainer Decorators

| Owner | Approvers |
| - | - |
| [Amin Saied](mailto:amsaied@microsoft.com) | [Krishan Subudhi](mailto:krkusuk@microsoft.com), [Jon Sleep](mailto:josleep@microsoft.com) |

Trainers handles the training logic in marlin as defined in the abstract `Trainer` class.
The basic implementation of this abstraction is the `SingleProcess` trainer, which - as
the name suggests - handles training on a single process (e.g. single GPU training).

## Motivation

As we add more trainers to the marlin library we may want to combine functionality from
existing trainers without duplicating logic. We require a design pattern that is flexible
and allows for responsibilities to be added to an object dynamically at run-time.

### Example

As a concrete example, marlin provides two additional trainers, `DDPTrainer` and
`SingleProcessAMP`, which handle distributed training and AMP-based training
respectively. We want to provide ability to choose DDPTrainer either with or without
AMP enabled. To avoid writing a new DDPTrainerWithAMP trainer, we propose the following
decorator design pattern - [Decorator pattern (wiki)](https://en.wikipedia.org/wiki/Decorator_pattern).

**Note.** This is not to be confused with Python decorators. 

## Decorator Pattern

We implement an `AbstractTrainerDecorator` class (which itself inherits from the
`Trainer` class). This class accepts a `Trainer` in its constructor,

```python
def __init__(self, trainer: Trainer):
    self.trainer = trainer  # trainer is a component of TrainerDecorator
```

and implements its interface by calling `self.train.method()` for any `method()`
the `Trainer` class requires.

### Example

To return to the concrete example, we create `DDPTrainer` as an
`AbstractTrainerDecorator`. This allows us to set up either:

```python
single_process_trainer = SingleProcess(...)
ddp_trainer = DDPTrainer(trainer=single_process_trainer)
```

or

```python
single_process_amp_trainer = SingleProcessAMP(...)
ddp_trainer_with_amp = DDPTrainer(trainer=single_process_amp_trainer)
```

## Concerns

One potential concern is that constructing trainers can become more complex with
this design by forcing the client to instantiate multiple trainers. We propose two
potential solutions:

1. Factory: Provide a factory method that abstracts the construction of the most
common trainers.
2. Add class methods to construct common trainers e.g. `DDPTrainer.from_single_process()`.