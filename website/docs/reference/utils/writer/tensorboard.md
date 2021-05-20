---
sidebar_label: tensorboard
title: utils.writer.tensorboard
---

Tensorboard writer module.

## Tensorboard Objects

```python
class Tensorboard(Writer)
```

This class implements a wrapper on Tensorboard&#x27;s SummaryWriter
for logging stats to Tensorboard. Please look here for detailed information
on each function:  https://pytorch.org/docs/stable/tensorboard.html#
Visit this blog for more examples on logging stats to Tensorboard:
https://krishansubudhi.github.io/deeplearning/2020/03/24/tensorboard-pytorch.html

#### log\_scalar

```python
 | log_scalar(k, v, step)
```

Log metric to Tensorboard graph.

#### log\_multi

```python
 | log_multi(k, v, step)
```

Log multiple metrics in the same Tensorboard graph.

#### log\_model

```python
 | log_model(flat_weights, flat_grads, step)
```

Log model weights and gradients to Tensorboard.

#### log\_embedding

```python
 | log_embedding(tag, mat, labels, step)
```

Log model embeddings to Tensorboard.

#### log\_graph

```python
 | log_graph(model, device)
```

Logs model graphs to Tensorboard.

**Arguments**:

- `model` _object_ - unwrapped model with a function get_sample_input() implemented.
- `device` _str_ - device type.

#### log\_image

```python
 | log_image(k, v, step, dataformats='HW')
```

Log image in Tensorboard.

#### log\_pr\_curve

```python
 | log_pr_curve(k, preds, labels, step)
```

Log Precision Recall curve in Tensorboard.

#### log\_args

```python
 | log_args(args)
```

Log all the Arguments used in the experiment to Tensorboard.

#### log\_histogram

```python
 | log_histogram(param_name, vals, step)
```

Log histograms in Tensorboard.
Avoid using small step size since it impact training time.

#### flush

```python
 | flush()
```

Flush the SummaryWriter to write out Summary to Tensorboard.

#### finish

```python
 | finish()
```

Flush the SummaryWriter to write out Summary to Tensorboard and
close SummaryWriter.

