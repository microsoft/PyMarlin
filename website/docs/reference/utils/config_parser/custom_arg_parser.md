---
sidebar_label: custom_arg_parser
title: utils.config_parser.custom_arg_parser
---

Custom Arguments Parser

## CustomArgParser Objects

```python
class CustomArgParser()
```

This class is part of utils and is provided to load arguments from the provided YAML config file.
Further, the default values of arguments from config file can be overridden via command line.
The class instance takes in the parser object and optional log_level.
This class needed to be instantiated in the main method inside the ELR_Scenario code.

&#x27;&#x27;Example for instantiation&#x27;&#x27;:
    parser = CustomArgParser()
    config = parser.parse()

The command line arguments to override default YAML config values are passed by adding a &#x27;.&#x27; between
namespace and the specific argument as shown in example below. If no namespace is present, then just
pass the argument name. All command line arguments are optional and need to be prefixed with &#x27;--&#x27;.
All commandline arguments not present in YAML config file will be ignored with a warning message.
Example commandline override:
python train.py --tmgr.epochs 4 --chkp.save_dir &quot;tmp\checkpoints&quot;

NOTE:
Supported types for CustomArgParser are int, float, str, lists. null is inferred implicitly as str.
If you intend to use other types, then please set a dummy default value in YAML file and pass the
intended value from commandline. Suggested defaults:
    str: null
    int: -1
    float: -1.0
    bool: pick either True or False
    list[int]: [-1, -1, -1]
    list[float] : [-1.0, -1.0, -1.0]

#### parse

```python
 | parse() -> Dict
```

Parse YAML config file, parse commandline arguments and merge the two
to get the final merged config dictionary.

Find and return the path of the file with greatest number of
completed epochs under dirpath (recursive search) for a given file
prefix, and optionally file extension.

**Arguments**:

  self
  

**Returns**:

- `self._config` _Dict_ - merged config dictionary containing all arguments.

