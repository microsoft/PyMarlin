---
sidebar_label: logging_utils
title: utils.logger.logging_utils
---

Logging util module

#### getlogger

```python
getlogger(name, log_level='INFO')
```

This method returns a logger object to be used by the calling class.
The logger object returned has the following format for all the logs:
&#x27;SystemLog: %(asctime)s:%(levelname)s : %(name)s : %(lineno)d : %(message)s&#x27;

**Arguments**:

- `name` _str_ - Directory under which to search for checkpointed files.
- `file_prefix` _str_ - Prefix to match for when searching for candidate files.
- `file_ext` _str, optional_ - File extension to consider when searching.
  

**Returns**:

- `logger` _object_ - logger object to use for logging.

