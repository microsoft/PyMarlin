# Checkpointing

The `pymarlin.utils,checkpointer.checkpoint_utils` module cointains the
 `AbstractCheckpointer` class that can be extended and
passed along to `pymarlin.core.trainer.Trainer` for checkpointing. A default implementation
is provided via `DefaultCheckpointer` in case no checkpointer is passed along to
`pymarlin.core.trainer.Trainer`.
Users can control the `DefaultCheckpointer` behavior via the
`DefaultCheckpointerArguments`,
and modify the arguments dataclass for their own checkpointers.
Here is an example of how to create your own checkpointer:
```python
    # Implement a dataclass for custom checkpointer arguments
    @dataclass
    class MyCheckpointerArguments:
        # Args for custom checkpointer class.
        checkpoint: bool = True
        save_dir: str = os.path.join(os.getcwd(), "checkpoints")
        model_state_save_dir: str = os.path.join(os.getcwd(), "checkpoints")
        load_dir: str = None
        load_filename: str = None
        file_prefix: str = "checkpoint"
        file_ext: str = "pt"
    # Implement AbstractCheckpointer
    class MyCheckpointer(AbstractCheckpointer):
        def __init__(self, args):
            # Initialize checkpointer by passing along args for user configurations
            # such as directory to load or save checkpoints, flags, etc.
            self.args = args
        def save(self, checkpoint_state, index, force=False):
            # Trainer calls this method at every epoch, regardless of any
            # arguments set, and thus this method should contain all logic for
            # how, where, and when a checkpoint needs to be saved when called. Users can
            # call any checkpointer's save method at other stages of the training lifecycle
            # with the provided hooks, and are encouraged to implement all
            # checkpointing logic here. An index argument is required, and can be
            # used to create a unique name for the file to be saved or as part of the
            # checkpointing logic. The optional force flag should allow disregarding
            # any custom logic implemented, so as to ensure Trainer can save
            # the last epoch when training and checkpointing is enabled.
            # Note that args can be customized to form save path in any way required
            # Where to save the checkpoint:
            path = os.path.join(self.args.save_dir,
                f"{self.args.file_prefix}_{index}.{self.args.file_ext}")
            # Custom logic for when to save a checkpoint here
            # When to save the checkpoint:
            if index % 5 == 0:
                # How to save the checkpoint:
                torch.save(checkpoint_state, path)
        def save_model(self, model_state, index):
            # Trainer will call this at the end of training.
            # An index argument is required to create a unique name for the file
            # to be saved.
            # Implement this method if you wish to save exclusively the ModuleInterface (model)
            # state at the end of training. As with save(), this is called automatically by
            # Trainer, but can be used at other stages of the training lifecycle via hooks.
            if self.args.checkpoint and self.args.model_state_save_dir:
                path = os.path.join(self.args.model_state_save_dir,
                    f"{self.args.file_prefix}_model_{index}.{self.args.file_ext}")
                torch.save(model_state, path)
        
        def load(self):
            # Implements logic to load a checkpointed file. Always called
            # upon initialization of Trainer. Leverage checkpointer args
            # to implement how and from where to load the checkpointed file, and
            # return the loaded checkpoint to Trainer. Trainer expects a Checkpoint
            # dataclass instance returned.
            if self.args.load_dir and self.args.load_filename:
                path = os.path.join(self.args.load_dir,
                                    self.args.load_filename)
                return torch.load(path, map_location=torch.device('cpu'))
            else:
                self.logger.warning('No checkpointer loaded, check load_fir and load_filename are set.')
                return Checkpoint()
    # Create instance of custom checkpointer
    my_args = MyCheckpointerArguments()
    my_checkpointer = MyCheckpointer(my_args)
    
    # Pass along custom checkpointer to Trainer
    trainer = Trainer(module=module_interface,
                      trainer_backend=trainer_backend,
                      args=trainer_args,
                      checkpointer=my_checkpointer)
```

Recall that these three methods are called automatically by `pymarlin.core.trainer.Trainer`:
* **load()**: at `pymarlin.core.trainer.Trainer` inicialization, before training.
* **save()**: at the end of every epoch and once more after training with force=True.
* **save_model()**: at the end of training.
Please review the `AbstractCheckpointer` documentation for precise method signatures for
correctly interfacing with `pymarlin.core.trainer.Trainer` if creating a custom checkpointer.
To customize *what* is checkpointed as a part of the attributes of `Checkpoint`, please
override the **get_state()** methods at
`ModuleInterface.get_state()`,
`Trainer.get_state()` and
`TrainerBackend.get_state()`.
For example, for `Trainer.get_state()`:
```python
    class MyTrainer(Trainer):
        def __init__(self, module, args, trainer_backend, checkpointer):
            super().__init__(module, args, trainer_backend, checkpointer)
        def get_state(self) -> dict:
            state_dict = {
                "last_epoch": self.last_epoch,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
                "args": self.args   # Adding something else we want to save
            }
            return state_dict
```
Please remember to also update **update_state()** methods if appropriate.
