import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertPreTrainedModel,
    BertLMPredictionHead,
    BertForSequenceClassification,
    BertForTokenClassification,
)

class TuringV1Model(BertModel):
    """Turing NLR V1 model without any task specific head.

    Example:

        >>> from marlin.models.turingv1.configuration_turingv1 import TuringV1Config
        >>> from marlin.models.turingv1.modeling_turingv1 import TuringV1Model
        >>> config = TuringV1Config()
        >>> model = TuringV1Model(config)

    Note:
        Full Turing model with heads cannot be loaded onto full Bert
        with heads due to mismatch in keys (self.cls.predictions vs
        self.predictions).

    """
    def __init__(self, config):
        super(TuringV1Model, self).__init__(config)

    def get_sample_input(self):
        return torch.ones(1, 128, dtype=torch.long)


class TuringV1ModelForPretraining(BertPreTrainedModel):
    """Turing NLR V1 model with pretraining MLM and SOP heads.

    Example:

        >>> from marlin.models.turingv1.configuration_turingv1 import TuringV1Config
        >>> from marlin.models.turingv1.modeling_turingv1 import TuringV1ModelForPretraining
        >>> config = TuringV1Config()
        >>> model = TuringV1ModelForPretraining(config)

    Note:
        Full Turing model with heads cannot be loaded onto full Bert
        with heads due to mismatch in keys (self.cls.predictions vs
        self.predictions).

    """
    def __init__(self, config):
        super(TuringV1ModelForPretraining, self).__init__(config)
        self.bert = TuringV1Model(config)
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 3)
        self.init_weights()
        self._tie_or_clone_weights(
            self.predictions.decoder, self.bert.embeddings.word_embeddings
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
        outputs = self.bert(
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
                seq_relationship_score.view(-1, 3), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

            outputs = (total_loss,) + outputs

        return outputs

    def encoder_state_dict(self, *args, **kwargs):
        return self.bert.state_dict(*args, **kwargs)

    def load_encoder_state_dict(self, model_state_dict):
        return self.bert.load_state_dict(model_state_dict)

    def get_sample_input(self):
        return torch.ones(1, 128, dtype=torch.long)


class TuringV1ForSequenceClassification(BertForSequenceClassification):
    """TuringV1ForSequenceClassification is fully compatible with Bert. The encoder
    was only trained on a slightly different pretraining task (SOP instead of NSP).
    """
    def __init__(self, config):
        super(TuringV1ForSequenceClassification, self).__init__(config)
        self.bert = TuringV1Model(config)
        self.init_weights()

class TuringV1ForTokenClassification(BertForTokenClassification):
    """TuringV1ForTokenClassification is fully compatible with Bert. The encoder
    was only trained on a slightly different pretraining task (SOP instead of NSP).
    """
    def __init__(self, config):
        super(TuringV1ForTokenClassification, self).__init__(config)
        self.bert = TuringV1Model(config)
        self.init_weights()