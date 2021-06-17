import os
from pymarlin.core import trainer

# too long import
from pymarlin.core.trainer_backend import build_trainer_backend
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser

from filelock import FileLock

# DeepSpeed + ORT
from deepspeed_methods.deepspeed_trainer import DeepSpeedTrainer
from deepspeed_methods.deepspeed_trainer_backend import DeepSpeedTrainerBackend, DeepSpeedDistributedTrainerBackend
from onnxruntime.training.ortmodule import ORTModule

from data import SummarizationData
from model_ortds import SummarizationBartModuleORT,SummarizationBartModuleDeepSpeedORT, SummarizationBartModule

if __name__ == '__main__':
    config = CustomArgParser(yaml_file_arg_key="config_path", default_yamlfile="config-ortds.yaml").parse()

    print(f"config: {config}")

    data = SummarizationData(root=config["data_path"])
    
    if config['ortds']:
        module_class = SummarizationBartModuleDeepSpeedORT
    elif config['ort']:
        module_class = SummarizationBartModuleORT
    else:
        module_class = SummarizationBartModule

    module = module_class(data, **config["module"], generate_kwargs=config["generate"])

    trainer_args = trainer.TrainerArguments(
        **config["trainer"],
        stats_args=trainer.stats.StatInitArguments(**config["stat"]),
        writer_args=trainer.WriterInitArguments(**config["wrt"]),
        checkpointer_args=trainer.DefaultCheckpointerArguments(**config["chkp"])
    )

    if config['ortds']:
        module.deepspeed_resume_from_checkpoint = config["chkp"]["load_dir"]
        module.deepspeed_ckpt_tag = config["module"]["deepspeed_ckpt_tag"]
        assert len(config["DEEPSPEED_CKPT_PREFIX"].strip()) > 0, f"config[\"DEEPSPEED_CKPT_PREFIX\"] must be non-empty"
        module.DEEPSPEED_CKPT_PREFIX = config["DEEPSPEED_CKPT_PREFIX"].strip()
        tr =  DeepSpeedDistributedTrainerBackend(DeepSpeedTrainerBackend()) if config["dist"] else DeepSpeedTrainerBackend()
        trainer = DeepSpeedTrainer(trainer_backend=tr, module=module, args=trainer_args)
    else:
        trainer = trainer.Trainer(module=module, args=trainer_args)

    trainer.train()
    trainer.validate()
