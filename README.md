# PyMarlin, a lightweight PyTorch library for agile deep learning!
[![Unit Tests](https://github.com/microsoft/PyMarlin/actions/workflows/test.yml/badge.svg)](https://github.com/microsoft/PyMarlin/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/microsoft/PyMarlin/branch/main/graph/badge.svg?token=wUF3ZODLpN)](https://codecov.io/gh/microsoft/PyMarlin)
[![Docs](https://github.com/microsoft/PyMarlin/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/microsoft/PyMarlin/actions/workflows/deploy-website.yml)
[![AzureML Canary](https://github.com/microsoft/PyMarlin/actions/workflows/canary.yml/badge.svg)](https://github.com/microsoft/PyMarlin/actions/workflows/canary.yml)
[![Test PyPi](https://github.com/microsoft/PyMarlin/actions/workflows/python-publish.yml/badge.svg)](https://github.com/microsoft/PyMarlin/actions/workflows/python-publish.yml)

PyMarlin was developed with the goal of simplifying the E2E Deep Learning experimentation lifecycle both for Microsoft Office data scientists. The library enables an agile way to quickly prototype a new AI scenario on dev box and seamlessly scale it training multi-node DDP GPU training with AzureML or other cloud services.

## Key features
- Provides public and enterprise **data pre-processing** recipes, which provides out of the box vanilla and parallel processing. It requires no additional code to run for AzureML or other environments easily.
- Provides **scalable model training** with support for Single Process, VM, multi-GPU, multi-node, distributed Data Parallel, mixed-precision (AMP, Apex) training. ORT and DeepSpeed based training are going to be available soon!
- (TBA) Provides out of the box **Plugins** that can be used for all typical NLP tasks like Sequence Classification, Named Entity Recognition and Seq2Seq text generation.
- Provides **reusable modules** for model checkpointing, stats collection, Tensorboard and compliant AML logging which can be customized based on your scenario.
- Provides **custom arguments parser** that allows for saving all the default values for arguments related to a scenario in an YAML config file, merging user provided arguments at runtime.
- PyMarlin is minimal and has a easy to understand codebase. PyMarlin was designed to make it easy for others to understand the entire codebase and customize according to their needs.

## Installation

    pip install pymarlin

Read the [installation doc](https://microsoft.github.io/PyMarlin/docs/installation) for more information.

## Start exploring!

### Full documentation website
Full website with [guides and SDK reference](https://microsoft.github.io/PyMarlin/).

### Train your first model with pymarlin
Check out the [CIFAR image classification example](hhttps://microsoft.github.io/PyMarlin/docs/examples/cifar).

### GLUE task benchmarking
Explore how to use pymarlin to [benchmark your language models on GLUE tasks](https://microsoft.github.io/PyMarlin/docs/examples/glue-tasks).

## We want your feedback!
Reach out to us with your [feedback and suggestions](https://microsoft.github.io/PyMarlin/docs/credits).
