---
sidebar_label: checkpoint_utils
title: utils.checkpointer.checkpoint_utils
---

Checkpointer class and utility functions.

Contains :class:`~.AbstractCheckpointer` class that can be extended and
passed along to :class:`~pymarlin.core.trainer.Trainer` for checkpointing. A default implementation
is provided via :class:`~.DefaultCheckpointer` in case no checkpointer is passed along to
:class:`~pymarlin.core.trainer.Trainer`.
Users can control the :class:`~.DefaultCheckpointer` behavior via the
:class:`~.DefaultCheckpointerArguments`,
and modify the arguments dataclass for their own checkpointers.

Here is an example of how to create your own checkpointer:

.. code-block:: python

# Implement a dataclass for custom checkpointer arguments
@dataclass
class MyCheckpointerArguments:
# Args for custom checkpointer class.
checkpoint: bool = True
save_dir: str = os.path.join(os.getcwd(), &quot;checkpoints&quot;)
model_state_save_dir: str = os.path.join(os.getcwd(), &quot;checkpoints&quot;)
load_dir: str = None
load_filename: str = None
file_prefix: str = &quot;checkpoint&quot;
file_ext: str = &quot;pt&quot;

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
# call any checkpointer&#x27;s save method at other stages of the training lifecycle
# with the provided hooks, and are encouraged to implement all
# checkpointing logic here. An index argument is required, and can be
# used to create a unique name for the file to be saved or as part of the
# checkpointing logic. The optional force flag should allow disregarding
# any custom logic implemented, so as to ensure Trainer can save
# the last epoch when training and checkpointing is enabled.

# Note that args can be customized to form save path in any way required
# Where to save the checkpoint:
path = os.path.join(self.args.save_dir,
f&quot;{self.args.file_prefix}_{index}.{self.args.file_ext}&quot;)

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
f&quot;{self.args.file_prefix}_model_{index}.{self.args.file_ext}&quot;)
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
return torch.load(path, map_location=torch.device(&#x27;cpu&#x27;))
else:
self.logger.warning(&#x27;No checkpointer loaded, check load_fir and load_filename are set.&#x27;)
return Checkpoint()

# Create instance of custom checkpointer
my_args = MyCheckpointerArguments()
my_checkpointer = MyCheckpointer(my_args)

# Pass along custom checkpointer to Trainer
trainer = Trainer(module=module_interface,
trainer_backend=trainer_backend,
args=trainer_args,
checkpointer=my_checkpointer)

Recall that these three methods are called automatically by :class:`~pymarlin.core.trainer.Trainer`:

* **load()**: at :class:`~pymarlin.core.trainer.Trainer` inicialization, before training.
* **save()**: at the end of every epoch and once more after training with force=True.
* **save_model()**: at the end of training.

Please review the :class:`~.AbstractCheckpointer` documentation for precise method signatures for
correctly interfacing with :class:`~pymarlin.core.trainer.Trainer` if creating a custom checkpointer.

To customize *what* is checkpointed as a part of the attributes of :class:`~.Checkpoint`, please
override the **get_state()** methods at
:func:`ModuleInterface.get_state() &lt;pymarlin.core.module_interface.ModuleInterface.get_state&gt;`,
:class:`Trainer.get_state() &lt;pymarlin.core.trainer.Trainer.get_state&gt;` and
:class:`TrainerBackend.get_state() &lt;pymarlin.core.trainer.TrainerBackend.get_state&gt;`.

For example, for :class:`Trainer.get_state() &lt;pymarlin.core.trainer.Trainer.get_state&gt;`:

.. code-block:: python

class MyTrainer(Trainer):
def __init__(self, module, args, trainer_backend, checkpointer):
super().__init__(module, args, trainer_backend, checkpointer)

def get_state(self) -&gt; dict:
state_dict = {
&quot;last_epoch&quot;: self.last_epoch,
&quot;optimizers&quot;: [optimizer.state_dict() for optimizer in self.optimizers],
&quot;schedulers&quot;: [scheduler.state_dict() for scheduler in self.schedulers],
&quot;args&quot;: self.args   # Adding something else we want to save
}
return state_dict

Please remember to also update **update_state()** methods if appropriate.

## DefaultCheckpointerArguments Objects

```python
@dataclass
class DefaultCheckpointerArguments()
```

Args for DefaultCheckpointer class.

## AbstractCheckpointer Objects

```python
class AbstractCheckpointer(ABC)
```

Abstract class for a checkpointer.

To create a custom checkpointer, users must implement the abstract
methods of this class and pass along an instance to ModuleInterface.
Custom checkpointers can be used at other stages of the
training lifecycle via callbacks.

#### save

```python
 | @abstractmethod
 | save(checkpoint_state: Checkpoint, index: int, force: Optional[bool]) -> None
```

Creates a checkpoint by saving a Checkpoint dataclass containing any
relevant states.

**Arguments**:

- `checkpoint_state` _Checkpoint_ - Checkpointed states.
- `index` _int_ - Using epoch as index is suggested.
- `force` _bool, optional_ - Saves checkpoint regardless of conditions
  if args.checkpoint is set to True. Used to always checkpoint
  models after the last epoch.

#### save\_model

```python
 | save_model(model_state: Dict, index: int) -> None
```

Creates a model checkpoint by saving model state.

**Arguments**:

- `model_state` _Dict_ - Model state as provided by ModuleInterface.
- `index` _int_ - Number to use to create a unique filename.
  Using epoch as index is suggested.

#### load

```python
 | @abstractmethod
 | load() -> Checkpoint
```

Load and return a checkpointed file.

Implements logic to load a checkpointed file as configured
via args used when constructing the checkpointer object. Always called
upon initialization of Trainer.

**Returns**:

- `Checkpoint` - Checkpointed states.

## DefaultCheckpointer Objects

```python
class DefaultCheckpointer(AbstractCheckpointer)
```

Default checkpointer implementation, implements AbstractCheckpointer and
contains a few helper functions for managing checkpointed files.

Must be initialized with DefaultCheckpointerArguments. The argument&#x27;s values
affect checkpointing as follows:

checkpoint: Flag indicating whether to checkpoint model when save()
    is called. Other conditions are implemented within save(), allowing this
    method to always be called within training loops and abstracting the
    checkpointing logic out of Trainer and implemented in this class.
delete_existing_checkpoints: Flag indicating whether to delete checkpoints
    under save_dir before training. New checkpoints are saved regardless.
period: Period of index at which to checkpoint model. Evaluates
    index % period == 0. This function is called with index set to the
    epoch, and thus checkpoints every &quot;period&quot; number of epochs. The last
    epoch is always checkpointed regardless.
save_dir: Path to directory where checkpoints are to be stored. Creates
    folder if it does not exist.
model_state_save_dir: Path to directory where checkpointed models are to
    be stored. Creates folder if it does not exist.
load_dir: Path to directory where checkpoints are to be loaded from.
    If not set, will not attempt to load a checkpoint. If load_filename
    is set, will search for this filename within the directory to load it.
    If load_filename is not set, will load the file via get_latest_file().
load_filename: Filename of checkpoint to load under load_dir, overrides
    automatic loading via get_latest_file().
file_prefix: Prefix of the checkpoint filename. Final filename to save
    will be {file_prefix}_{index}.{file_ext}, or in the case of saving with
    save_model(), {file_prefix}_mode_{index}.{file_ext}.
file_ext: File extension for the checkpoint filename when saving and when
    searching under load_dir for loading via get_latest_file().
    When cleaning save_dir via delete_existing_checkpoints=True, only files
    with this extension are considered.
log_level: Logging level for checkpointer module (Default: &#x27;INFO&#x27;).

#### \_\_init\_\_

```python
 | __init__(args: DefaultCheckpointerArguments)
```

Initialize checkpointer and delete existing checkpointed files
under save_dir if delete_existing_checkpoints is set to True.

#### save

```python
 | save(checkpoint_state: Checkpoint, index: int, force=False) -> str
```

Creates a checkpoint by saving a Checkpoint dataclass containing any
relevant states as a python Dict.

Evaluates conditions and, if met, saves a provided dataclass
which should contain any states that users require to save as
part of a checkpoint under args.save_dir. An additional index
argument is required to create a unique name for the file to be
saved. The optional force flag will disregard conditions other
than the checkpoint flag that enables this behavior. The condition
for saving with DefaultCheckpointer is index being a multiple
of the args.period.

**Arguments**:

- `checkpoint_state` _Checkpoint_ - Checkpointed states.
- `index` _int_ - Number to use to create a unique filename and
  evaluate conditions for checkpointing. Using epoch as index
  is suggested.
- `force` _bool, optional_ - Saves checkpoint regardless of conditions
  if args.checkpoint is set to True. Used to always checkpoint
  states after the last epoch.
  

**Returns**:

- `str` - Path to checkpointed file.

#### save\_model

```python
 | save_model(model_state: Dict, index: int) -> str
```

Checkpoints a model state, leveraging torch.save().

Evaluates if checkpointing is enabled and if a model save directory
has been set, and saves a provided model state. An additional index
argument is required to create a unique name for the file to be
saved.

**Arguments**:

- `model_state` _Dict_ - Model state as provided by ModuleInterface.
- `index` _int_ - Number to use to create a unique filename.
  Using epoch as index is suggested.

**Returns**:

- `str` - Path to checkpointed file.

#### load

```python
 | load() -> Checkpoint
```

Attempt to load and return a checkpointed file leveraging torch.load().
The checkpoined file is assumed to be created with save() and thus be
a python Dict.

This method is always called upon initialization of Trainer.
Searches for and attempts to load a checkpointed file based on
args. If no load_dir is set, returns None. If a load_dir and
load_filename have been set, the file &quot;load_filename&quot; under
load_dir is directly loaded (the filename must include extension).
If only load_dir is set, get_latest_file() is called to seach the
folder for the file with the largest integer (index) in its filename,
and returns that path for loading.

**Returns**:

- `Checkpoint` - Checkpointed states.

#### get\_latest\_file

```python
 | @staticmethod
 | get_latest_file(load_dir: str, file_prefix: str, file_ext: str = 'pt', logger=getlogger(__name__)) -> str
```

Get the path to the last checkpointed file.

Find and return the path of the file with greatest number of
completed epochs under dirpath (recursive search) for a given file
prefix, and optionally file extension.

**Arguments**:

- `load_dir` _str_ - Directory under which to search for
  checkpointed files.
- `file_prefix` _str_ - Prefix to match for when searching
  for candidate files.
- `file_ext` _str, optional_ - File extension to consider
  when searching.
  

**Returns**:

- `str` - Path to latest checkpointed file.

#### check\_mk\_dir

```python
 | check_mk_dir(dirpath: str) -> None
```

Check if the path exists, and if it doesn&#x27;t creates it.

**Arguments**:

- `dirpath` _str_ - Directory under which to search for
  checkpointed files.

