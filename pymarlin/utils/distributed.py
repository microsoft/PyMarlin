"""distributed utils"""
import os
from dataclasses import dataclass
from typing import Optional
from functools import wraps
from azureml.core.run import Run
import torch

@dataclass
class DistributedTrainingArguments:
    local_rank: int = 0
    global_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    init_method: str = "env://"
    gather_frequency: Optional[int] = None

@dataclass
class DistributedPreprocessArguments:
    local_rank: int = 0
    global_rank: int = 0
    world_size: int = 1
    node_count: int = 1
    local_size: int = 1
    node_rank: Optional[int] = None

class SequentialDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, seed=0, drop_last=False, **kwargs):
        super().__init__(dataset, shuffle=False, num_replicas=num_replicas, rank=rank, seed=seed, drop_last=drop_last, **kwargs)

def ranks_already_set(args) -> bool:
    """Return True is both local and global ranks have been set."""
    is_local_rank_set = args.local_rank > -1
    is_global_rank_set = args.global_rank > -1
    return is_local_rank_set and is_global_rank_set

def fetch_ranks_from_azureml_preprocess():
    """Look up distributed arguments from Azure ML environment variables.

    Assumes OpenMPI image.

    Note:
        Sets up NCCL environment variables used by Azure ML:

        - NCCL_SOCKET_IFNAME
        - NCCL_IB_DISABLE
    """
    ranks = DistributedPreprocessArguments()

    run = Run.get_context()
    run.get_status()
    ranks.node_count = run.get_details()['runDefinition']['nodeCount']
    ranks.local_size = run.get_details()['runDefinition']['mpi']['processCountPerNode']

    ranks.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
    ranks.global_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
    ranks.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))

    return ranks

def fetch_ranks_from_azureml():
    """Look up distributed arguments from Azure ML environment variables.

    Assumes OpenMPI image.

    Note:
        Sets up NCCL environment variables used by Azure ML:

        - NCCL_SOCKET_IFNAME
        - NCCL_IB_DISABLE
    """
    ranks = DistributedTrainingArguments()
    ranks.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
    ranks.global_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
    ranks.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
    return ranks


def fetch_ranks_from_torch_distributed_launch():
    """Read distributed arguments set by torch.distributed.launch via environment variables."""
    ranks = DistributedTrainingArguments()
    ranks.local_rank = int(os.environ["LOCAL_RANK"])
    ranks.global_rank = int(os.environ["RANK"])
    ranks.world_size = int(os.environ["WORLD_SIZE"])
    return ranks


def set_environment_variables_for_nccl_backend():
    """Sets distributed training environments for azureml openmpi runs with NCCL backend."""

    # NCCL environment. Still works without it.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_IB_DISABLE"] = "0"  # for IB

    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(
        os.environ["OMPI_COMM_WORLD_SIZE"]
    )

    if single_node:
        master_node = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        master_port = "54965"
    else:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")

        master_node = master_node_params[0]
        master_port = (
            os.environ["MASTER_PORT"] if "MASTER_PORT" in os.environ else "6105"
        )

    # set env variables
    os.environ["MASTER_ADDR"] = master_node
    os.environ["MASTER_PORT"] = master_port

def rank_zero_only(fn):
    """Decorates functions to only execute on global rank 0, else wait via torch.distributed"""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            res = fn(*args, **kwargs)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return res
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    return wrapped_fn
rank_zero_only.rank = 0 # by default
