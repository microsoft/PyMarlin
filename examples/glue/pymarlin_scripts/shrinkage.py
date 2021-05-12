from random import sample
import copy

import torch
import torch.nn as nn

from transformers.modeling_utils import prune_linear_layer

from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__)

def modifyEncoderWidth(new_encoder, classifier_layer, config):
    modify_prop = round(config.num_student_atthead / config.num_attention_heads, 2)
    new_size = int(config.hidden_size / config.num_attention_heads) * config.num_student_atthead
    if modify_prop > 1:
        logger.info(f"Model width increase by {modify_prop}x.")
        hidden_keep_index = None
    else:
        logger.info(f"Model width shrinking by {modify_prop}x.")
        hidden_keep_index = torch.tensor(sample(list(range(0, config.hidden_size)), new_size)).long()

    new_encoder = _modifyEmbeddingsWidth(new_encoder, hidden_keep_index, new_size)
    new_encoder, hidden_keep_index = _modifyEncodingWidth(new_encoder, hidden_keep_index, config)
    new_encoder, hidden_keep_index = _modifyPoolerWidth(new_encoder, hidden_keep_index, config)
    if hidden_keep_index is None:
        new_classifier_layer = extend_linear_layer(classifier_layer, new_size = new_size, dim = 1)
    else:
        new_classifier_layer = prune_linear_layer(classifier_layer, index = hidden_keep_index, dim = 1)
    return new_encoder, new_classifier_layer


def _modifyEmbeddingsWidth(model, hidden_keep_index, new_size):
    new_model = copy.deepcopy(model)
    new_embed = new_model.embeddings
    if hidden_keep_index is None:
        new_embed.word_embeddings = extend_embedding_layer(new_embed.word_embeddings, new_size)
        new_embed.position_embeddings = extend_embedding_layer(new_embed.position_embeddings, new_size)
        if hasattr(new_embed, 'token_type_embeddings'):
            new_embed.token_type_embeddings = extend_embedding_layer(new_embed.token_type_embeddings, new_size)
        if hasattr(new_embed, 'LayerNorm'):
            new_embed.LayerNorm = extend_layernorm(new_embed.LayerNorm, new_size)
    else:
        new_embed.word_embeddings = prune_embedding_layer(new_embed.word_embeddings, hidden_keep_index)
        new_embed.position_embeddings = prune_embedding_layer(new_embed.position_embeddings, hidden_keep_index)
        if hasattr(new_embed, 'token_type_embeddings'):
            new_embed.token_type_embeddings = prune_embedding_layer(new_embed.token_type_embeddings, hidden_keep_index)
        if hasattr(new_embed, 'LayerNorm'):
            new_embed.LayerNorm = prune_layernorm(new_embed.LayerNorm, hidden_keep_index, dim=0)
    del model
    return new_model

def _modifyEncodingWidth(model, hidden_keep_index, config):
    oldModuleList = model.encoder.layer
    newModuleList = torch.nn.ModuleList()
    for index, layer in enumerate(oldModuleList):
        new_layer, hidden_keep_index = _modifyBertLayerWidth(layer, hidden_keep_index, config)
        newModuleList.append(new_layer)
    new_model = copy.deepcopy(model)
    new_model.encoder.layer = newModuleList
    del model
    return new_model, hidden_keep_index

def _modifyBertLayerWidth(layer, hidden_keep_index, config):
    layer_components = ['attention', 'intermediate', 'output']
    new_layer = copy.deepcopy(layer)
    new_hidden_size = int(int(config.hidden_size / config.num_attention_heads) * config.num_student_atthead)
    new_intermediate_size = config.student_intermediate_size
    for component in layer_components:
        if component == 'attention':
            if hidden_keep_index is None:
                new_layer.attention.self = extend_BertSelfAttention_head(new_layer.attention.self, new_heads_size = config.num_student_atthead, new_hidden_size = new_hidden_size)
                new_layer.attention.output.LayerNorm = extend_layernorm(new_layer.attention.output.LayerNorm, new_size= new_hidden_size)
                new_layer.attention.output.dense = extend_linear_layer(new_layer.attention.output.dense, new_size= new_hidden_size, dim = 1)
                new_layer.attention.output.dense = extend_linear_layer(new_layer.attention.output.dense, new_size= new_hidden_size, dim = 0)
            else:
                attention_head_drop_index = sample(list(range(0,config.num_attention_heads)), config.num_attention_heads - config.num_student_atthead) #currently only random pruning
                new_layer.attention.prune_heads(attention_head_drop_index)
                new_layer.attention.self.query = prune_linear_layer(new_layer.attention.self.query, index = hidden_keep_index, dim = 1)
                new_layer.attention.self.key = prune_linear_layer(new_layer.attention.self.key, index = hidden_keep_index, dim = 1)
                new_layer.attention.self.value = prune_linear_layer(new_layer.attention.self.value, index = hidden_keep_index, dim = 1)
                hidden_keep_index = torch.tensor(sample(list(range(0, config.hidden_size)), new_hidden_size)).long()
                new_layer.attention.output.dense = prune_linear_layer(new_layer.attention.output.dense, index = hidden_keep_index, dim = 0)
                new_layer.attention.output.LayerNorm = prune_layernorm(new_layer.attention.output.LayerNorm, index = hidden_keep_index, dim = 0)
        elif component == 'intermediate':
            if hidden_keep_index is None:
                new_layer.intermediate.dense = extend_linear_layer(new_layer.intermediate.dense, new_size= new_hidden_size, dim = 1)
                new_layer.intermediate.dense = extend_linear_layer(new_layer.intermediate.dense, new_size= new_intermediate_size, dim = 0)

            else:
                new_layer.intermediate.dense = prune_linear_layer(new_layer.intermediate.dense, index = hidden_keep_index, dim = 1)
                hidden_keep_index = torch.tensor(sample(list(range(0, config.intermediate_size)), new_intermediate_size)).long()
                new_layer.intermediate.dense = prune_linear_layer(new_layer.intermediate.dense, index = hidden_keep_index, dim = 0)

        else:
            if hidden_keep_index is None:
                new_layer.output.dense = extend_linear_layer(new_layer.output.dense, new_size= new_intermediate_size, dim = 1)
                new_layer.output.dense = extend_linear_layer(new_layer.output.dense, new_size= new_hidden_size, dim = 0)
                new_layer.output.LayerNorm = extend_layernorm(new_layer.output.LayerNorm, new_size = new_hidden_size)

            else:
                new_layer.output.dense = prune_linear_layer(new_layer.output.dense, index = hidden_keep_index, dim = 1)
                hidden_keep_index = torch.tensor(sample(list(range(0, config.hidden_size)), new_hidden_size)).long()
                new_layer.output.dense = prune_linear_layer(new_layer.output.dense, index = hidden_keep_index, dim = 0)
                new_layer.output.LayerNorm = prune_layernorm(new_layer.output.LayerNorm, index = hidden_keep_index, dim = 0)
    del layer
    return new_layer, hidden_keep_index

def _modifyPoolerWidth(model, hidden_keep_index, config):
    new_hidden_size = int(int(config.hidden_size / config.num_attention_heads) * config.num_student_atthead)
    new_model = copy.deepcopy(model)
    new_pooler = new_model.pooler
    if hidden_keep_index is None:
        new_pooler.dense = extend_linear_layer(new_pooler.dense, new_size = new_hidden_size, dim = 1)
        new_pooler.dense = extend_linear_layer(new_pooler.dense, new_size = new_hidden_size, dim = 0)
    else:
        new_pooler.dense = prune_linear_layer(new_pooler.dense, index = hidden_keep_index, dim = 1)
        hidden_keep_index = torch.tensor(sample(list(range(0, config.hidden_size)), new_hidden_size)).long()
        new_pooler.dense = prune_linear_layer(new_pooler.dense, index = hidden_keep_index, dim = 0)
    del model
    return new_model, hidden_keep_index

def prune_embedding_layer(layer, index):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(1, index).clone().detach()
    new_size = list(layer.weight.size())
    new_size[1] = len(index)
    new_layer = nn.Embedding(layer.num_embeddings, new_size[1], padding_idx = layer.padding_idx).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    return new_layer

def prune_layernorm(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.LayerNorm(new_size[dim], eps = layer.eps).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def extend_BertSelfAttention_head(layer, new_heads_size, new_hidden_size):
    layer.query = extend_linear_layer(layer.query, new_size= new_hidden_size, dim = 1)
    layer.query = extend_linear_layer(layer.query, new_size= new_hidden_size, dim = 0)
    layer.key = extend_linear_layer(layer.key, new_size= new_hidden_size, dim = 1)
    layer.key = extend_linear_layer(layer.key, new_size= new_hidden_size, dim = 0)
    layer.value = extend_linear_layer(layer.value, new_size= new_hidden_size, dim = 1)
    layer.value = extend_linear_layer(layer.value, new_size= new_hidden_size, dim = 0)
    layer.num_attention_heads = new_heads_size
    layer.all_head_size = new_hidden_size
    return layer

def extend_linear_layer(layer, new_size, dim = 0):
    """ Extend a linear layer (a model parameters) by the new size along the dim specified.
        Return the extended layer as a new layer with requires_grad=True.
    """
    W = layer.weight.clone().detach()
    size = list(layer.weight.size())
    index = torch.tensor([x for x in range(size[dim])]).to(layer.weight.device)
    if layer.bias is not None:
        b = layer.bias.clone().detach()
        index_bias = torch.tensor([x for x in range(size[0])]).to(layer.weight.device)
    size[dim] = new_size
    new_layer = nn.Linear(size[1], size[0], bias = layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.index_copy_(dim, index, W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.index_copy_(0, index_bias, b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def extend_embedding_layer(layer, new_size):
    """ Extend a embedding layer (a model parameters) by the new size along the dim specified.
        Return the extended layer as a new layer with requires_grad=True.
    """
    W = layer.weight.clone().detach()
    size = list(layer.weight.size())
    index = torch.tensor([x for x in range(size[1])]).to(layer.weight.device)
    new_layer = nn.Embedding(layer.num_embeddings, new_size, padding_idx = layer.padding_idx).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.index_copy_(1, index, W.contiguous())
    new_layer.weight.requires_grad = True
    return new_layer

def extend_layernorm(layer, new_size):
    """ Extend a layernorm layer (a model parameters) by the new size along the dim specified.
        Return the extended layer as a new layer with requires_grad=True
    """
    W = layer.weight.clone().detach()
    index = torch.tensor([x for x in range(layer.normalized_shape[0])]).to(layer.weight.device)
    if layer.bias is not None:
        b = layer.bias.clone().detach()
    new_layer = nn.LayerNorm(new_size, eps = layer.eps).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.index_copy_(0, index, W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.index_copy_(0, index, b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer
