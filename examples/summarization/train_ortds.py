import os
from pymarlin.core import trainer_backend, module_interface, trainer

# too long import
from pymarlin.core.trainer_backend import build_trainer_backend
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser

from filelock import FileLock

from deepspeed_methods.deepspeed_trainer import deepspeed_Trainer
from deepspeed_methods.deepspeed_trainer_backend import deepspeed_trainer_backend, deepspeed_dist_trainer_backend

from data import SummarizationData
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

print(f"config: {config}")

dm = SummarizationData()
dm.setup_datasets(root=config["data_path"])

tm = SummarizationBartModule_ds_ort(dm, **config["module"], generate_kwargs=config["generate"])

tmArgs = trainer.TrainerArguments(
    **config["trainer"],
    stats_args=trainer.stats.StatInitArguments(**config["stat"]),
    writer_args=trainer.WriterInitArguments(**config["wrt"]),
    checkpointer_args=trainer.DefaultCheckpointerArguments(**config["chkp"])
)

if config["module"]["deepspeed"]:

    tm.deepspeed_resume_from_checkpoint = config["chkp"]["load_dir"]
    tm.deepspeed_ckpt_tag = config["module"]["deepspeed_ckpt_tag"]
    assert len(config["DEEPSPEED_CKPT_PREFIX"].strip()) > 0, f"config[\"DEEPSPEED_CKPT_PREFIX\"] must be non-empty"
    tm.DEEPSPEED_CKPT_PREFIX = config["DEEPSPEED_CKPT_PREFIX"].strip()
    tr =  deepspeed_dist_trainer_backend() if config["dist"] else deepspeed_trainer_backend()
    trainer = deepspeed_Trainer(trainer_backend=tr, module=tm, args=tmArgs)
else:
    trainer = trainer.Trainer(module=tm, args=tmArgs)

trainer.train()
trainer.validate()
