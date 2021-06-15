# Getting Started

### Welcome to PyMarlin, a lightweight PyTorch library for agile deep learning!
PyMarlin was developed with the goal of simplifying the E2E Deep Learning experimentation lifecycle for data scientists. The library enables an agile way to quickly prototype a new AI scenario on your dev box and seamlessly scale to multi-node GPU training in AzureML or any other cloud services.

## Key features
- Provides public and enterprise **data pre-processing** recipes, which provides out of the box vanilla and parallel processing. It requires no additional code to run for AzureML or other environments easily.
- Provides **scalable model training** with support for Single Process, VM, multi-GPU, multi-node, distributed Data Parallel, mixed-precision (AMP, Apex) training. ORT and DeepSpeed based training are going to be available soon!
- Provides out of the box **Plugins** that can be used for all typical NLP tasks like Sequence Classification, Named Entity Recognition and Seq2Seq text generation.
- Provides **reusable modules** for model checkpointing, stats collection, Tensorboard and compliant AML logging which can be customized based on your scenario.
- Provides **custom arguments parser** that allows for saving all the default values for arguments related to a scenario in an YAML config file, merging user provided arguments at runtime.
- All core modules are thoroughly **linted**,**unit tested** and even ran E2E (multi-node, GPU) in AzureML.
- PyMarlin is minimal and has a easy to understand codebase. PyMarlin was designed to make it easy for others to understand the entire codebase and customize according to their needs.

## Start exploring!

### Train your first model with pymarlin

Check out [CIFAR image classification](examples/cifar.md) from the EXAMPLES section.

### GLUE task benchmarking

Explore how to use pymarlin to [benchmark your models on GLUE tasks](examples/glue-tasks.md).

## We want your feedback!

Reach out to us with your [feedback and suggestions](credits.md).