---
sidebar_label: distributed
title: utils.distributed
---

#### ranks\_already\_set

```python
ranks_already_set(args) -> bool
```

Return True is both local and global ranks have been set.

#### fetch\_ranks\_from\_azureml\_preprocess

```python
fetch_ranks_from_azureml_preprocess()
```

Look up distributed arguments from Azure ML environment variables.

Assumes OpenMPI image.

**Notes**:

  Sets up NCCL environment variables used by Azure ML:
  
  - NCCL_SOCKET_IFNAME
  - NCCL_IB_DISABLE

#### fetch\_ranks\_from\_azureml

```python
fetch_ranks_from_azureml()
```

Look up distributed arguments from Azure ML environment variables.

Assumes OpenMPI image.

**Notes**:

  Sets up NCCL environment variables used by Azure ML:
  
  - NCCL_SOCKET_IFNAME
  - NCCL_IB_DISABLE

#### fetch\_ranks\_from\_torch\_distributed\_launch

```python
fetch_ranks_from_torch_distributed_launch()
```

Read distributed arguments set by torch.distributed.launch via environment variables.

#### set\_environment\_variables\_for\_nccl\_backend

```python
set_environment_variables_for_nccl_backend()
```

Sets distributed training environments for azureml openmpi runs with NCCL backend.

#### rank\_zero\_only

```python
rank_zero_only(fn)
```

Decorates functions to only execute on global rank 0, else wait via torch.distributed

