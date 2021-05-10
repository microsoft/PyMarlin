---
sidebar_label: stdout
title: utils.writer.stdout
---

Stdout writer module.

## Stdout Objects

```python
class Stdout(Writer)
```

This class implements the stdout writer for stats.

#### log\_scalar

```python
 | log_scalar(k, v, step)
```

Log metric to stdout.

#### log\_multi

```python
 | log_multi(k, v, step)
```

Log metric to stdout.

#### log\_model

```python
 | log_model(flat_weights, flat_grads, step)
```

Log model to stdout.
Can slow down training. Only use for debugging.
It&#x27;s logged in Tensorboard by default.

#### log\_graph

```python
 | log_graph(model, device=None)
```

Log model graph to stdout.

