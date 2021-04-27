import math
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention, BertIntermediate, \
    BertPreTrainedModel, BertPredictionHeadTransform, BertLMPredictionHead

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
    #test
    test = BertLayerNorm(1024, eps=1e-5)

except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm as BertLayerNorm


## Create a Bert Pre Layer Norm + FP32 Operation based model here
def gelu_fp32(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN_fp32 = {"gelu": gelu_fp32, "relu": torch.nn.functional.relu, "swish": swish}

class BertEmbeddings_noLN(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings_noLN, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, tenant_ids  = None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddings_Tenant_noLN(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings_Tenant_noLN, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.tenant_embeddings = nn.Embedding(config.max_tenants, config.hidden_size)
        #self.tenant_embeddings.initialize_first_zero = True
        self.tenant_as_token = False
        if hasattr(config, 'tenant_as_token'):
            self.tenant_as_token = config.tenant_as_token
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, tenant_ids = None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if tenant_ids is None:
            if self.tenant_as_token:
                tenant_ids = torch.zeros((input_ids.shape[0],1), dtype=torch.long, device=input_ids.device)
            else:
                tenant_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)     

        assert self.tenant_embeddings.num_embeddings >= tenant_ids.max().data
        tenant_embeddings = self.tenant_embeddings(tenant_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.tenant_as_token :
            embeddings = torch.cat((embeddings, tenant_embeddings), 1).to(input_ids.device)            
            return embeddings
        else:            
            embeddings += tenant_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings        

class BertSelfAttention_fp32(BertSelfAttention):
    def __init__(self, config):
        super(BertSelfAttention_fp32, self).__init__(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        pdtype = attention_scores.dtype
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores.float()).to(pdtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput_preLN(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput_preLN, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertAttention_preLN(nn.Module):
    def __init__(self, config):
        super(BertAttention_preLN, self).__init__()
        self.self = BertSelfAttention_fp32(config)
        self.output = BertSelfOutput_preLN(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate_fp32(BertIntermediate):
    def __init__(self, config):
        super(BertIntermediate_fp32, self).__init__(config)
        self.intermediate_act_fn = ACT2FN_fp32[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act


class BertOutput_preLN(nn.Module):
    def __init__(self, config):
        super(BertOutput_preLN, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states   

class BertLayer_preLN(nn.Module):
    def __init__(self, config):
        super(BertLayer_preLN, self).__init__()
        self.attention = BertAttention_preLN(config)
        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate_fp32(config)
        self.output = BertOutput_preLN(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        input_layer_norm = self.PreAttentionLayerNorm(hidden_states)
        attention_outputs = self.attention(input_layer_norm, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        intermediate_input = hidden_states + attention_output

        intermediate_layer_norm = self.PostAttentionLayerNorm(intermediate_input)
        intermediate_output = self.intermediate(intermediate_layer_norm)

        layer_output = self.output(intermediate_output)        
        outputs = (layer_output + intermediate_input,) + attention_outputs[1:]  # add attentions if we output them 
        return outputs


class BertEncoder_preLN(nn.Module):
    def __init__(self, config):
        super(BertEncoder_preLN, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer = nn.ModuleList([BertLayer_preLN(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, tenant_ids = None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i]
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        last_layer_hidden_states_norm = self.FinalLayerNorm(hidden_states)
        outputs = (last_layer_hidden_states_norm,)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertEncoder_Tenant_preLN(nn.Module):
    """ 
    Bert encoder variant which allows us to add tenant token as part of sequence starting from a layer set by the argument --tenant_tok_layer
    """
    def __init__(self, config):
        super(BertEncoder_Tenant_preLN, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer = nn.ModuleList([BertLayer_preLN(config) for _ in range(config.num_hidden_layers)])
        self.tenant_as_token = False
        self.tenant_tok_layer = 0
        if hasattr(config, 'tenant_as_token'):
            self.tenant_as_token = config.tenant_as_token
        if hasattr(config, 'tenant_tok_layer'):
            self.tenant_tok_layer = config.tenant_tok_layer

    def forward(self, hidden_states, attention_mask=None, head_mask=None, tenant_ids = None):
        all_hidden_states = ()
        all_attentions = ()
        # remove last token from hidden layer, 
        if self.tenant_as_token and self.tenant_tok_layer != 0:
            tenant_emb = hidden_states[:, -1, :] # removing tenant tokens
            hidden_states = hidden_states[:, :-1, :]
            attention_mask = attention_mask[:, :, :, :-1]

        for i, layer_module in enumerate(self.layer):            
            if i > 0 and i == self.tenant_tok_layer:                
                #print("Setting tenant token at layer: ", i, " extending attention mask")                
                hidden_states = torch.cat((hidden_states, tenant_emb.unsqueeze(1)), dim = 1)
                extend_mask = torch.ones((attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2],1), dtype=attention_mask.dtype, device=hidden_states.device)
                attention_mask = torch.cat((attention_mask, extend_mask), dim =3)
            
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i]
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        last_layer_hidden_states_norm = self.FinalLayerNorm(hidden_states)
        outputs = (last_layer_hidden_states_norm,)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertModel_preLN(BertPreTrainedModel):
    '''
    Pre-LN reference: https://openreview.net/pdf?id=B1x8anVFPr
    '''
    def __init__(self, config):
        super(BertModel_preLN, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings_noLN(config)
        self.encoder = BertEncoder_preLN(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            num_layers = self.config.num_hidden_layers
            std = self.config.initializer_range
            if hasattr(module, 'bert_output_layer'):
                std = self.config.initializer_range / math.sqrt(2.0 * num_layers)
            module.weight.data.normal_(mean=0.0, std=std)

            if hasattr(module, 'initialize_first_zero'):
                module.weight[0] = torch.zeros_like(module.weight[0])

            
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, tenant_ids = None):
        if attention_mask is None:
            if hasattr(self.config, 'tenant_as_token') and self.config.tenant_as_token:            
                attention_mask = torch.cat((torch.ones_like(input_ids), torch.ones((input_ids.shape[0],1), dtype=torch.long, device=input_ids.device)), 1)
            else:
                attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, tenant_ids = tenant_ids)      
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask,
                                       tenant_ids = tenant_ids) # make change in all encoder variants
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            
        if hasattr(self.embeddings, 'tenant_embeddings'):
            for param in self.embeddings.tenant_embeddings.parameters():
                param.requires_grad = True

class BertPredictionHeadTransform_fp32(BertPredictionHeadTransform):
    def __init__(self, config):
        super(BertPredictionHeadTransform_fp32, self).__init__(config)
        self.transform_act_fn = ACT2FN_fp32[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act


class BertLMPredictionHead_fp32(BertLMPredictionHead):
    def __init__(self, config):
        super(BertLMPredictionHead_fp32, self).__init__(config)
        self.transform = BertPredictionHeadTransform_fp32(config)

