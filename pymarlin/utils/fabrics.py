"""Compute fabric specific utility methods."""
import os
import importlib.util


def is_azureml_mpirun() -> bool:
    """Check if run set up by azureml using OpenMPI image.

    When running MPIRUN with OpenMPI images, AzureML sets a specific combination
    of environment variables which we check for here, specifically::

        OMPI_COMM_WORLD_RANK  # the rank of the process
        OMPI_COMM_WORLD_SIZE  # the world size
        OMPI_COMM_WORLD_LOCAL_RANK  # the local rank of the process on the node
        OMPI_COMM_WORLD_LOCAL_SIZE  # number of processes on the node

    and one of the following::

        AZ_BATCH_MASTER_NODE  # multiple nodes
        AZ_BATCHAI_MPI_MASTER_NODE  # single node
    """
    is_openmpi_image: bool = (
        "OMPI_COMM_WORLD_RANK" in os.environ
        and "OMPI_COMM_WORLD_SIZE" in os.environ
        and "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
        and "OMPI_COMM_WORLD_LOCAL_SIZE" in os.environ
    )

    is_azureml_mpirun_env: bool = (
        "AZ_BATCH_MASTER_NODE" in os.environ
        or "AZ_BATCHAI_MPI_MASTER_NODE" in os.environ
    )

    return bool(is_openmpi_image and is_azureml_mpirun_env)


def is_torch_distributed_launch_via_environment_variables() -> bool:
    """Check if torch.distributed.launch used to submit the job using environment variables."""

    env_vars = os.environ
    is_using_environment_vars: bool = (
        "RANK" in env_vars
        and "MASTER_ADDR" in env_vars
        and "MASTER_PORT" in env_vars
        and "WORLD_SIZE" in env_vars
    )

    return is_using_environment_vars


def is_azureml_run_with_sdk() -> bool:
    """Check if we are running on Azure ML with azureml-sdk."""
    if not _is_azureml_available():
        print("Unable to import azureml sdk.")
        return False

    import azureml.core.run

    run = azureml.core.run.Run.get_context()
    is_azureml_run = False

    try:
        run.get_status()
        is_azureml_run = True
    except AttributeError:
        print("This is not an Azure ML run")

    return is_azureml_run


def _is_azureml_available() -> bool:
    """Check sys.modules to see if azureml.core.run is available.
    See https://github.com/huggingface/transformers/blob/02e05fb0a532e572b56ba75dad6ba3db625bbdeb/src/transformers/integrations.py#L81
    """
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None
