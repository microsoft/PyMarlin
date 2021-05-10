---
sidebar_label: fabrics
title: utils.fabrics
---

Compute fabric specific utility methods.

#### is\_azureml\_mpirun

```python
is_azureml_mpirun() -> bool
```

Check if run set up by azureml using OpenMPI image.

When running MPIRUN with OpenMPI images, AzureML sets a specific combination
of environment variables which we check for here, specifically::

    OMPI_COMM_WORLD_RANK  # the rank of the process
    OMPI_COMM_WORLD_SIZE  # the world size
    OMPI_COMM_WORLD_LOCAL_RANK  # the local rank of the process on the node
    OMPI_COMM_WORLD_LOCAL_SIZE  # number of processes on the node

and one of the following::

    AZ_BATCH_MASTER_NODE  # multiple nodes
    AZ_BATCHAI_MPI_MASTER_NODE  # single node

#### is\_torch\_distributed\_launch\_via\_environment\_variables

```python
is_torch_distributed_launch_via_environment_variables() -> bool
```

Check if torch.distributed.launch used to submit the job using environment variables.

#### is\_azureml\_run\_with\_sdk

```python
is_azureml_run_with_sdk() -> bool
```

Check if we are running on Azure ML with azureml-sdk.

