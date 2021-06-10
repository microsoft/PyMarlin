import os
from pymarlin.core import trainer_backend, module_interface, trainer

# too long import
from pymarlin.core.trainer_backend import build_trainer_backend
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser

from data import SummarizationData

from filelock import FileLock

from deepspeed.deepspeed_trainer import deepspeed_Trainer
from deepspeed.deepspeed_trainer_backend import deepspeed_trainer_backend, deepspeed_dist_trainer_backend
from model import SummarizationBartModule_ds_ort

try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


config = CustomArgParser(yaml_file_arg_key="config_path").parse()

if config["dist"] and config["AML"]:
    try:
        config["cuda"] = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        config["world_size"] = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    except KeyError as e:
        print(f"Encountered KeyError: {e}, which likely result from one of following: (1) Not using AML (2) Using AML with 1 node and 1 process")
        print(f"Setting config[\"cuda\"] to 0")
        config["cuda"] = 0

print(f"config: {config}")

dm = SummarizationData()
dm.setup_datasets(root=config["data_path"])

summarization_module = SummarizationBartModule_ds_ort(dm, **config["tm"], generate_kwargs=config["generate"])

trainer_args = trainer.TrainerArguments(
    **config["tmgr"],
    stats_args=trainer.stats.StatInitArguments(**config["stat"]),
    writer_args=trainer.WriterInitArguments(**config["wrt"]),
    checkpointer_args=trainer.DefaultCheckpointerArguments(**config["chkp"])
)

if config["module"]["deepspeed"]:
    summarization_module.deepspeed_resume_from_checkpoint = config["chkp"]["load_dir"]
    summarization_module.deepspeed_ckpt_tag = config["chkp"]["deepspeed_ckpt_tag"]
    assert len(config["DEEPSPEED_CKPT_PREFIX"].strip()) > 0, f"config[\"DEEPSPEED_CKPT_PREFIX\"] must be non-empty"
    summarization_module.DEEPSPEED_CKPT_PREFIX = config["DEEPSPEED_CKPT_PREFIX"].strip()

trainer = trainer.Trainer(module=summarization_module, args=trainer_args)

trainer.train()
trainer.validate()
