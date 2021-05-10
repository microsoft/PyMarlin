---
sidebar_label: aml
title: utils.writer.aml
---

AML writer module.

## Aml Objects

```python
class Aml(Writer)
```

This class implements the Azure ML writer for stats.

#### log\_scalar

```python
 | log_scalar(k, v, step)
```

Log metric to AML.

#### log\_multi

```python
 | log_multi(k, v, step)
```

Log metrics to stdout.

