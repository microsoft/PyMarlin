'''pymarlin.plugins.hfdistill_utils'''
from pymarlin.core.module_interface import ModuleInterface
from transformers import AutoConfig
from pymarlin.utils.distributed import rank_zero_only
from pymarlin.utils.logger.logging_utils import getlogger

logger = getlogger(__name__, "DEBUG")

import os
import copy
import dataclasses
from dataclasses import field

import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize, softmax, log_softmax

@dataclasses.dataclass
class DistillationArguments:
    enable: bool = False
    # config_output_dir: str = None
    student_model_config_path: str = None
    student_model_config_file: str = None
    student_model_path: str = None
    student_model_file: str = None
    student_layers: list = field(default_factory=lambda: [0 - 6 - 11])
    loss_types: list = field(
        default_factory=lambda: ["logits"]
    )  # logits, labels, attentions, hidden_states
    loss_weights: list = field(default_factory=lambda: [1])
    temperature: float = 1


def build_distill_module(base):
    class DistillHfModule(base):
        """ModuleInterface which does knowledge distillation for the task performed in parent class ModuleInterface.
        Both student and teacher architectures must be Huggingface transformers (albeit student is a smaller version).
        `setup_model` overrides the `model` attribute to contain the student model instead of the
        teacher. The teacher is set in `teacher` attribute.
        `train_step` is overridden to do teacher and student forward pass, and computes various types of
        losses such as logits_loss, representation_loss (using attentions, hidden_states, embeddings).
        Representation loss can only be computed if student and teacher have comparable dimensions.

        Assumptions:
            Expects the parent class's model to be defined in the `model` attribute.
            Expects the parent class's data interface to be defined in the `data` attribute.
            `data` should have a method `get_labels()` which returns a list of labels.
            Expects the parent class to have a `_setup_config` method which creates and
            returns a Huggingface config object.

        base (ModuleInterface): ModuleInterface class which does teacher training for the specific task.
        """

        def __init__(self, distill_args, *module_args):
            super().__init__(*module_args)
            self.distill_args = distill_args
            logger.info(f"Initialized distillation module...")

        def setup_model(self, automodel_class):
            """Initializes student and teacher model weights.
            student: `DistillHfModule.model`
            teacher: `DistillHfModule.teacher`
            For teacher initialization:
                Option 1: Load weights from specified files mentioned in YAML config
                        model:
                            model_config_path
                            model_config_file
                            model_path
                            model_file
                Option 2: Load from Huggingface model hub, specify string in YAML config as:
                        model:
                            hf_model
            For student initialization:
                distill:
                    student_model_config_path
                    student_model_config_file
                    student_model_path
                    student_model_file
            If above args are not provided, then student is the same architecture as teacher.
            Both student and teacher models must be Huggingface transformers.

            Args:
                automodel_class: Huggingface AutoModel class (ForSeq, ForToken, etc.)
            """
            # setup teacher model
            if self.args.model_args.model_path is not None:
                self.teacher_config = self._setup_config()
                logger.info(f"Model filename: {self.args.model_args.model_file}")
                self.teacher = automodel_class.from_pretrained(
                    os.path.join(
                        self.args.model_args.model_path, self.args.model_args.model_file
                    ),
                    config=self.teacher_config,
                )
            else:
                self.teacher_config = AutoConfig.from_pretrained(
                    self.args.model_args.hf_model
                )
                self.teacher_config.num_labels = len(self.data.get_labels())
                self.teacher = automodel_class.from_pretrained(
                    self.args.model_args.hf_model, config=self.teacher_config
                )
            # setup student model
            if self.distill_args.student_model_path is not None:
                self.student_config = self._setup_config()
                self.model = automodel_class.from_pretrained(
                    os.path.join(
                        self.distill_args.student_model_path,
                        self.distill_args.student_model_file,
                    ),
                    config=self.student_config,
                )
            else:  # same arch as teacher
                self.model = copy.deepcopy(self.teacher)
                self.student_config = copy.deepcopy(self.teacher_config)
            if len(self.distill_args.student_layers) > 0:
                layer_modules = getattr(
                    self.model, self.args.model_args.encoder_key
                ).encoder.layer
                new_layer_modules = extract_layers(
                    layer_modules, self.distill_args.student_layers
                )
                getattr(
                    self.model, self.args.model_args.encoder_key
                ).encoder.layer = new_layer_modules
                self.student_config.num_hidden_layers = len(
                    self.distill_args.student_layers
                )
            self.teacher.eval()
            self.output_hidden = (
                True if "hidden_states" in self.distill_args.loss_types else False
            )
            self.output_attentions = (
                True if "attentions" in self.distill_args.loss_types else False
            )

        def _inputs_to_device(self, batch, device):
            inputs = {}
            for k, v in batch.items():
                if v is not None:
                    inputs[k] = v.to(device)
            return inputs

        def train_step(self, global_step, batch, device):
            self.teacher.eval()
            inputs = self._inputs_to_device(batch, device)
            teacher_outputs = self.teacher.forward(
                **inputs,
                output_hidden_states=self.output_hidden,
                output_attentions=self.output_attentions,
            )  # label_loss, logits, hidden, attns
            student_outputs = self.model.forward(
                **inputs,
                output_hidden_states=self.output_hidden,
                output_attentions=self.output_attentions,
            )
            total_loss = torch.zeros([1], dtype=student_outputs[0].dtype, device=device)
            for i, k in enumerate(self.distill_args.loss_types):
                if k == "labels":
                    student_scores = student_outputs.loss
                    teacher_scores = teacher_outputs.loss
                else:
                    student_scores = getattr(student_outputs, k)
                    teacher_scores = getattr(teacher_outputs, k)

                if student_scores is not None and teacher_scores is not None:
                    if k == "logits":
                        total_loss += self.distill_args.loss_weights[i] * logits_loss(
                            student_scores,
                            teacher_scores,
                            temperature=self.distill_args.temperature,
                        )
                    else:
                        total_loss += self.distill_args.loss_weights[
                            i
                        ] * representations_loss(
                            student_scores,
                            teacher_scores,
                            [*range(len(self.distill_args.student_layers))],
                            self.distill_args.student_layers,
                        )
            logger.debug(f"Loss = {total_loss.item()}")
            return total_loss

        # @rank_zero_only
        # def on_end_train(self, global_step):
        # logger.info(f"Finished training. Saving student model and config to {self.distill_args.config_output_dir}.")
        # os.makedirs(self.distill_args.config_output_dir, exist_ok=True)
        # self.student_config.to_json_file(os.path.join(self.distill_args.config_output_dir, self.args.model_args.model_config_file), use_diff=False)

    return DistillHfModule


def extract_layers(oldModuleList, layer_ids):
    newModuleList = torch.nn.ModuleList()
    logger.info(f"Extracting layers {layer_ids}")
    for i in layer_ids:
        newModuleList.append(oldModuleList[i])
    return newModuleList


# logits: 1 x logits -> logits
def logits_loss(
    student_inputs,
    teacher_inputs,
    temperature=1,
    reduction="batchmean",
    scale_loss_by_temp=False,
):
    loss_function = nn.KLDivLoss(reduction=reduction)
    loss = loss_function(
        input=log_softmax(student_inputs / temperature, dim=-1),
        target=softmax(teacher_inputs / temperature, dim=-1),
    )
    if scale_loss_by_temp:
        loss = loss * (temperature ** 2)
    return loss


# embs: (layer0 only) bs x seqlen x hid (no concat or transformation) -> loss*hid
# attns: layers x bs x heads x seqlen x seqlen -> bs*layers x heads x seqlen**2 -> loss*seqlen
# hids: layers x bs x seqlen x hid -> bs*layers x seqlen x hid -> loss*hid
def representations_loss(
    student_inputs,
    teacher_inputs,
    student_layer_ids,
    teacher_layer_ids,
    loss_function=None,
    reduction="mean",
    normalize=True,
    scale_loss_by_rep=True,
):
    if loss_function is None:
        loss_function = nn.MSELoss(reduction=reduction)
    normalized_student = prepare_inputs(student_inputs, student_layer_ids, normalize)
    normalized_teacher = prepare_inputs(teacher_inputs, teacher_layer_ids, normalize)
    if normalized_teacher.size()[-1] != normalized_student.size()[-1]:
        W = nn.Linear(
            normalized_teacher.size()[-1], normalized_student.size()[-1]
        )  # wt matrix to transform teacher hid size to student
        normalized_teacher = W(normalized_teacher)
    loss = loss_function(normalized_student, normalized_teacher)
    if scale_loss_by_rep:
        loss = loss * normalized_student.size()[-1]
    return loss


def prepare_inputs(inputs, layer_ids, do_normalize=True):
    inputs = concat_layers(inputs, layer_ids)
    inputs = inputs.view(inputs.size()[0], inputs.size()[1], -1)
    if do_normalize:
        inputs = normalize(inputs, dim=-1)
    return inputs


def concat_layers(layers, layer_ids):
    concatenated_layers = layers[0]
    for i in range(1, len(layer_ids)):
        concatenated_layers = torch.cat((concatenated_layers, layers[layer_ids[i]]))
    return concatenated_layers
