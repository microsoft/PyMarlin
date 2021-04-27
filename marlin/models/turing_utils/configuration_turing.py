from transformers.configuration_utils import PretrainedConfig

class TuringConfig(PretrainedConfig):
    """Configuration class for Turing models."""

    def __init__(
        self,
        attention_probs_dropout_prob=0.1,
        finetuning_task=None,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        torchscript=False,
        type_vocab_size=2,
        vocab_size=30522,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.finetuning_task = finetuning_task
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_labels = num_labels
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.torchscript = torchscript
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
