import torch
from torch import nn
from torch.nn.functional import normalize, softmax, log_softmax

from shrinkage import modifyEncoderWidth
from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__)

def extract_layers(oldModuleList, layer_ids):
    newModuleList = torch.nn.ModuleList()
    logger.info(f"Extracting layers {layer_ids}")
    for i in layer_ids:
        newModuleList.append(oldModuleList[i])
    return newModuleList

def logits_loss(student_inputs, teacher_inputs, temperature=1, reduction='batchmean', scale_loss_by_temp=False):
    loss_function = nn.KLDivLoss(reduction=reduction)
    loss = loss_function(
        input=log_softmax(student_inputs/temperature, dim=-1),
        target=softmax(teacher_inputs/temperature, dim=-1)
    )
    if scale_loss_by_temp:
        loss = loss * (temperature**2)
    return loss

# emb: (layer0 only) bs x seqlen x hid (no concat or transformation) -> loss*hid -- why only first layer hid is considered emb?
# attn: layers x bs x heads x seqlen x seqlen -> bs*layers x heads x seqlen**2 -> loss*seqlen (or is it seqlen**2)?
# hidden: layers x bs x seqlen x hid -> bs*layers x seqlen x hid -> loss*hid -- why does teacher use layernum+1 and student start from layer1?
# logit: 1 x logits -> logits
def representations_loss(student_inputs, teacher_inputs, student_layer_ids, teacher_layer_ids, loss_function=None, reduction='mean', normalize=True, scale_loss_by_rep=True):
    if loss_function is None:
        loss_function = nn.MSELoss(reduction=reduction)
    normalized_student = prepare_inputs(student_inputs, student_layer_ids, normalize)
    normalized_teacher = prepare_inputs(teacher_inputs, teacher_layer_ids, normalize)
    if normalized_teacher.size()[-1] != normalized_student.size()[-1]:
        W = nn.Linear(normalized_teacher.size()[-1], normalized_student.size()[-1]) # wt matrix to transform teacher hid size to student
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
