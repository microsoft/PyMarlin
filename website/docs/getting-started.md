# Getting Started

### Welcome to PyMarlin, a lightweight PyTorch library for agile deep learning!
PyMarlin is a lightweight PyTorch extension library for agile deep learning experimentation. PyMarlin was developed with the goal of simplifying the E2E Deep Learning experimentation lifecycle for data scientists. The library enables an agile way to quickly prototype a new AI scenario on your dev box and seamlessly scale to multi-node GPU training in AzureML or any other cloud services.

## Key features
- **Data pre-processing** module which enables data preprocessing recipes to scale from single CPU to multi-CPU and multi node. 
- **Infra-agnostic design**: native Azure ML integration implies the same code running on local dev-box can also run directly on any VM or Azure ML cluster.
- **Trainer backend abstraction** with support for Single Process (CPU/GPU), distributed Data Parallel, mixed-precision (AMP, Apex) training. ORT and Deepspeed libraries are also integrated to get the best distributed training throughputs.
- Out-of-the-box **Plugins** that can be used for typical NLP tasks like Sequence Classification, Named Entity Recognition and Seq2Seq text generation.
- **Utility modules** for model checkpointing, stats collection and Tensorboard events logging which can be customized based on your scenario.
- **Custom arguments parser** that allows for saving all the default values for arguments related to a scenario in a YAML config file, merging user supplied arguments at runtime.


## Start exploring!

### Train your first model with pymarlin

Check out [CIFAR image classification](examples/cifar.md) from the EXAMPLES section.

### GLUE task benchmarking

Explore how to use pymarlin to [benchmark your models on GLUE tasks](examples/glue-tasks.md).

## We want your feedback!

Reach out to us with your [feedback and suggestions](credits.md).