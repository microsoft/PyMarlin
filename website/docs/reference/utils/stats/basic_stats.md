---
sidebar_label: basic_stats
title: utils.stats.basic_stats
---

Basic Stats module.

## StatInitArguments Objects

```python
@dataclasses.dataclass
class StatInitArguments()
```

Stats Arguments.

## BasicStats Objects

```python
class BasicStats()
```

Basis Stats class provides a common place for collects long interval stats and step interval
stats that can be recorded in the various writers provided at the time of calling rebuild()
in trainer. This class is used as a Singleton pattern via global_stats provided in the
__init__.py file.

#### rebuild

```python
 | rebuild(args: StatInitArguments, writers: Iterable)
```

Rebuild Stat Args and Writers.

#### reset

```python
 | reset()
```

Reset all stats.

#### reset\_short

```python
 | reset_short()
```

Reset step interval stats.

#### reset\_long

```python
 | reset_long()
```

Reset long interval stats.

#### update

```python
 | update(k, v, frequent=False)
```

Update step interval and long interval scalar stats.

#### update\_multi

```python
 | update_multi(k, v: dict, frequent=False)
```

Update step interval and long interval multiple scalar stats.

#### update\_matplotlib\_figure

```python
 | update_matplotlib_figure(fig, tag)
```

Update matplotlib figure.

#### update\_image

```python
 | update_image(k, v, dataformats='HW')
```

Update image.
Will be logged with infrequent metric.

#### update\_pr

```python
 | update_pr(k, preds, labels)
```

Update pr curve stats.
Only binary classification
preds = probabilities

#### update\_histogram

```python
 | update_histogram(k, vals, extend=False)
```

Update histogram stats.

#### update\_embedding

```python
 | update_embedding(k, embs, labels)
```

Update embeddings.
 Used to project embeddings with corresponding labels (numerical).

#### update\_system\_stats

```python
 | update_system_stats()
```

Update system stats related to Memory and Compute (CPU and GPUs) usage.

#### log\_long\_stats

```python
 | log_long_stats(step)
```

Log long interval stats to correponding writers.

#### log\_args

```python
 | log_args(args)
```

Log Arguments to correponding writers.

#### log\_model

```python
 | log_model(step, model, force=False, grad_scale=1)
```

Log model to correponding writers.

#### log\_graph

```python
 | log_graph(model, device)
```

Log graph to correponding writers.

#### finish

```python
 | finish()
```

Call finish() on all writers.

