# coding=utf-8
""" TuringV3 model configuration """
from transformers.configuration_utils import PretrainedConfig

class TuringV3Config(PretrainedConfig):
    """Configuration class to store the configuration of a `BertModel`.
      """

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        # num_labels=2,
        # output_attentions=False,
        # output_hidden_states=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        task_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        rel_pos_type=0, # however 2 is used in both pretraining and certain finetuning tasks
        max_rel_pos=128,
        rel_pos_bins=32,
        fast_qkv=False,
        initializer_range=0.02,
        vocab_size=30522,
        **kwargs
    ):
        """Constructs TuringV3Config.

            Args:
                vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
                hidden_size: Size of the encoder layers and the pooler layer.
                num_hidden_layers: Number of hidden layers in the Transformer encoder.
                num_attention_heads: Number of attention heads for each attention layer in
                    the Transformer encoder.
                intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                    layer in the Transformer encoder.
                hidden_act: The non-linear activation function (function or string) in the
                    encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
                hidden_dropout_prob: The dropout probabilitiy for all fully connected
                    layers in the embeddings, encoder, and pooler.
                attention_probs_dropout_prob: The dropout ratio for the attention
                    probabilities.
                max_position_embeddings: The maximum sequence length that this model might
                    ever be used with. Typically set this to something large just in case
                    (e.g., 512 or 1024 or 2048).
                type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                    `BertModel`.
                initializer_range: The sttdev of the truncated_normal_initializer for
                    initializing all weight matrices.
            """
        super(TuringV3Config, self).__init__(**kwargs)

        self.model_type = "turingv3"

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        # self.num_labels = num_labels
        # self.output_attentions = output_attentions
        # self.output_hidden_states = output_hidden_states
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.task_dropout_prob = task_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.rel_pos_type = rel_pos_type
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv