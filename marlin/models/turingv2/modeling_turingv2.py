import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertForSequenceClassification,
    BertForTokenClassification,
)

from ..turing_utils.modeling_utils import (
    BertModel_preLN,
    BertLMPredictionHead_fp32,
)


class TuringV2Model(BertModel_preLN):
    def __init__(self, config):
        super(TuringV2Model, self).__init__(config)

    def get_sample_input(self):
        return torch.ones(1, 128, dtype=torch.long)


class TuringV2ModelForPretraining(BertPreTrainedModel):
    def __init__(self, config):
        super(TuringV2ModelForPretraining, self).__init__(config)
        self.encoder = TuringV2Model(config)
        self.predictions = BertLMPredictionHead_fp32(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.init_weights()
        self._tie_or_clone_weights(
            self.predictions.decoder, self.encoder.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)

        # add hidden states and attention if they are here
        outputs = (prediction_scores, seq_relationship_score) + outputs[2:]
        
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

            outputs = (total_loss,) + outputs

        return outputs

    def encoder_state_dict(self, *args, **kwargs):
        return self.encoder.state_dict(*args, **kwargs)

    def load_encoder_state_dict(self, model_state_dict):
        return self.encoder.load_state_dict(model_state_dict)

    def get_sample_input(self):
        return torch.ones(1, 128, dtype=torch.long)

    def load_state_dict(self, state_dict, strict=False):
        incmp_keys = super().load_state_dict(state_dict, strict)
        return incmp_keys

class TuringV2ForSequenceClassification(BertForSequenceClassification):
    """TuringV2ForSequenceClassification is the same as BertForSequenceClassification
    except that it uses the TuringV2Model encoder instead of BertModel.
    The V2 encoder was trained with pre-layer normalization.
    """
    def __init__(self, config):
        super(TuringV2ForSequenceClassification, self).__init__(config)
        self.bert = TuringV2Model(config)
        self.init_weights()

class TuringV2ForTokenClassification(BertForTokenClassification):
    """TuringV2ForTokenClassification is the same as BertForTokenClassification
    except that it uses the TuringV2Model encoder instead of BertModel.
    The V2 encoder was trained with pre-layer normalization.
    """
    def __init__(self, config):
        super(TuringV2ForTokenClassification, self).__init__(config)
        self.bert = TuringV2Model(config)
        self.init_weights()