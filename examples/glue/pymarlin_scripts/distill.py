import os
import sys
import dataclasses
import argparse
import copy

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification, AutoConfig

from pymarlin.core import trainer, trainer_backend, module_interface
from pymarlin.utils.stats import global_stats
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser
from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__)

from data import TaskData, DataInterfaceArguments, InputFeatures
from train import Recipe
from init_args import ModuleInterfaceArguments, ModelArgs, DistillArgs
import distill_utils

class DistillRecipe(Recipe):
    def __init__(self, args, distill_args, datamodule):
        super().__init__(args, datamodule)
        self.distill_args = distill_args
        self.loss_types = [str(item) for item in self.distill_args.loss_types.strip('[]').split('-')]
        self.loss_weights = [float(item) for item in self.distill_args.loss_weights.strip('[]').split('-')]
        self.student_layers = [int(item) for item in self.distill_args.student_layers.strip('[]').split('-')]

    def setup_models(self):
        self._setup_configs()
        # teacher setup
        if self.args.model_args.get_latest_ckpt:
            filenames = [f for f in os.listdir(self.args.model_args.model_wts_path) if (f.endswith('.pt') and not f.endswith('ort.pt')) or f.endswith('.bin')]
            self.args.model_args.model_file = max(filenames)
            logger.info(f"Model filename from get_latest_ckpt: {self.args.model_args.model_file}")
        self.teacher = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.args.model_args.model_wts_path, self.args.model_args.model_file),
            config=self.model_config
            )
        # student setup
        self.model = copy.deepcopy(self.teacher)
        if len(self.student_layers) > 0:
            layer_modules = getattr(self.model, self.args.model_args.encoder_key).encoder.layer
            new_layer_modules = distill_utils.extract_layers(layer_modules, self.student_layers)
            getattr(self.model, self.args.model_args.encoder_key).encoder.layer = new_layer_modules
        
        if self.distill_args.width_shrinkage != 0:
            def modify_config(model_config, distill_args):
                model_config.num_student_atthead = int(np.ceil(distill_args.width_shrinkage * model_config.num_attention_heads))
                hidden_size_multiplier = model_config.hidden_size / model_config.num_attention_heads
                model_config.student_hidden_size = int(model_config.num_student_atthead * hidden_size_multiplier)
                model_config.student_intermediate_size = int(model_config.intermediate_size * distill_args.width_shrinkage)
                return model_config
            self.model_config = modify_config(self.model_config, self.distill_args)
            new_encoder, new_cls_layer = distill_utils.modifyEncoderWidth(
                                            getattr(self.model, self.args.model_args.encoder_key),
                                            self.model.classifier,
                                            self.model_config
                                            )
            setattr(self.model, self.args.model_args.encoder_key, new_encoder)
            setattr(self.model, 'classifier', new_cls_layer)       
        self.teacher.eval()
        self.output_hidden = True if 'hidden_states' in self.loss_types else False
        self.output_attentions = True if 'attentions' in self.loss_types else False
        return (self.model, self.teacher)

    def train_step(self, global_step, batch, device):
        self.teacher.eval()
        inputs = self._inputs_to_device(batch, device)
        teacher_outputs = self.teacher.forward(**inputs,
                            output_hidden_states=self.output_hidden,
                            output_attentions=self.output_attentions,
                            ) # label_loss, logits, hidden, attns
        student_outputs = self.model.forward(**inputs,
                            output_hidden_states=self.output_hidden,
                            output_attentions=self.output_attentions,
                            )
        total_loss = torch.zeros([1], dtype=student_outputs[0].dtype, device=device)
        for i, k in enumerate(self.loss_types):
            if k == 'labels':
                student_scores = student_outputs.loss
                teacher_scores = teacher_outputs.loss
            else:
                student_scores = getattr(student_outputs, k)
                teacher_scores = getattr(teacher_outputs, k)

            if student_scores is not None and teacher_scores is not None:
                if k == 'logits':
                    total_loss += self.loss_weights[i] * distill_utils.logits_loss(
                        student_scores, teacher_scores,
                        temperature=self.distill_args.temperature,
                    )
                elif k != 'logits' and self.distill_args.width_shrinkage == 0:
                    total_loss += self.loss_weights[i] * distill_utils.representations_loss(
                                    student_scores,
                                    teacher_scores,
                                    [*range(len(self.student_layers))],
                                    self.student_layers
                    )
        return total_loss

    def _cleanup_config(self):
        if len(self.student_layers) > 0:
            self.model_config.num_hidden_layers = len(self.student_layers)
        if self.distill_args.width_shrinkage != 0:
            self.model_config.hidden_size = self.model_config.student_hidden_size
            self.model_config.num_attention_heads = self.model_config.num_student_atthead
            self.model_config.intermediate_size = self.model_config.student_intermediate_size
            del self.model_config.student_hidden_size
            del self.model_config.num_student_atthead
            del self.model_config.student_intermediate_size

    def on_end_train(self, global_step):
        logger.info(f"Finished training. Saving model and config to {self.args.output_dir}.")
        torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, self.args.model_args.model_file))
        self._cleanup_config()
        self.model_config.to_json_file(os.path.join(self.args.output_dir, self.args.model_args.model_config_file), use_diff=False)


if __name__ == "__main__":

    parser = CustomArgParser(log_level='DEBUG')
    config = parser.parse()
    logger.info(f"final merged config = {config}\n")

    os.makedirs(config['tmod']['output_dir'], exist_ok=True)

    data = TaskData(DataInterfaceArguments(**config['dmod']))

    model_args = ModelArgs(**config['model'])
    config['tmod']['model_args'] = model_args
    recipe_args = ModuleInterfaceArguments(**config['tmod'])
    distill_args = DistillArgs(**config['distill'])
    recipe = DistillRecipe(recipe_args, distill_args, data)
    student, teacher = recipe.setup_models()
    tmgr_args = trainer.TrainerArguments(
        **config["tmgr"],
        stats_args=trainer.stats.StatInitArguments(**config['stats']),
        writer_args=trainer.WriterInitArguments(**config['wrts']),
        checkpointer_args=trainer.DefaultCheckpointerArguments(**config['ckpt'])
    )

    trainer_backend_backend = trainer_backend.SingleProcess()
    if recipe_args.trainer_backend != 'SingleProcess':
        trainer_backend_backend = getattr(trainer_backend, recipe_args.trainer_backend)(trainer_backend_backend)
    trainer = trainer.Trainer(trainer_backend=trainer_backend_backend, module=recipe, args=tmgr_args)
    getattr(trainer, recipe_args.operation)()

# train mode
# python distill.py --tmod.trainpath "processed_data\RTE\train" --tmod.valpath "processed_data\RTE\dev" --tmod.output_dir "distill_out" --model.model_wts_path "C:\Users\shgullap\Desktop\workspace\models\bert-base-wiki-ckpt" --model.model_config_path "C:\Users\shgullap\Desktop\workspace\models\bert-base-wiki-ckpt"
# to infer distilled model, run train.py with trainer.validate()
# python train.py --tmod.valpath "processed_data\RTE\dev" --tmod.output_dir "distill_dev_out" --model.model_wts_path "distill_out" --model.model_config_path "distill_out" 