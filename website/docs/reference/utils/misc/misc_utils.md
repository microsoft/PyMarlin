---
sidebar_label: misc_utils
title: utils.misc.misc_utils
---

Miscellaneous utility functions

#### snake2camel

```python
snake2camel(name)
```

This method changes input name from snake format to camel format.

**Arguments**:

- `name` _str_ - snake format input name.
  

**Returns**:

- `name` _str_ - camel format input name.

#### clear\_dir

```python
clear_dir(path, skips=None)
```

This method deletes the contents of the directory for which path
has been provided and not included in the skips list.

**Arguments**:

- `path` _str_ - Path for directory to be deleted.
- `skips` _List[str]_ - List of paths for sub directories to be skipped from deleting.

#### debug

```python
debug(method)
```

This method wraps input method with debug calls to measure time taken for
the given input method to finish.

**Arguments**:

- `method` _function_ - Method which needs to be timed.
  

**Returns**:

- `debugged` _method_ - debugged function.

