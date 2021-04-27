# MarlinAutoModels

Marlin extends the Huggingface
[`AutoModel`](https://huggingface.co/transformers/model_doc/auto.html?highlight=from_pretrained#automodel)
class to include the Turing family.

## Available Models

- Turing Version 1:
    - `turingv1-base-uncased`: 12 layers
    - `turingv1-large-uncased`: 24 layers
    - `turingv1-mini-uncased`: 2 layers
- Turing Version 2:
    - `turingv2-base-uncased`: 12 layers
    - `turingv2-large-uncased`: 24 layers
    - `turingv2-mini-uncased`: 2 layers
- Turing Version 3:
    - `turingv3-base-uncased`: 12 layers
    - `turingv3-large-uncased`: 24 layers
    - `turingv3-mini-uncased`: 2 layers


## Example: AutoModel support for Turing-family

### Load a randomly initialized instance of Turing model

```python
from marlin.models.auto.configuration_auto import MarlinAutoConfig
from marlin.models.auto.modeling_auto import MarlinAutoModel
config = MarlinAutoConfig.from_pretrained("turingv3-base-uncased")
model = MarlinAutoModel.from_config(config)
```

### Load a pretrained instance of Turing NLR Version 1

Note: At this time we only support loading pretrained models from a `state_dict`.

```python
from marlin.models.auto.modeling_auto import MarlinAutoModel
config = MarlinAutoConfig.from_pretrained("turingv3-large-uncased")
model = MarlinAutoModel.from_pretrained(
    pretrained_model_name_or_path="turingv3_state_dict_model.pt",
    config=config,
)
```

Which will return a pretrained model with weights specified in "turingv3_state_dict_model.pt":

    TuringV3Model(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 128)
        (position_embeddings): Embedding(512, 128)
        (token_type_embeddings): Embedding(2, 128)
        (LayerNorm): BertLayerNorm()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=128, out_features=128, bias=True)
                (key): Linear(in_features=128, out_features=128, bias=True)
                (value): Linear(in_features=128, out_features=128, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
      ...

Here `pretrained_model_name_or_path` can either be a file storing the pretrained states, or it can be in memory.

## Manual Turing Models

The `MarlinAutoModel` and `MarlinAutoConfig` classes provide convenient ways to
automatically load models from the Turing family from one of the _pre-configured_
options provided (`turingv[1|2|3]-[mini|base|large]-uncased`).

In addition it is possible to manually configure the Turing models. For example:

```python
from marlin.models.turingv1.configuration_turingv1 import TuringV1Config
from marlin.models.turingv1.modeling_turingv1 import TuringV1
config = TuringV1Config(num_attention_heads=5, num_hidden_layers=2)
model = TuringV1Model(config)
```

In this case the full TuringV1Config will be as follows:

```python
TuringV1Config {
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "turingv1",
  "num_attention_heads": 5,
  "num_hidden_layers": 2,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

The same manual configuration is possible with Version 2 and Version 3.

#### Turing Version 2

```python
from marlin.models.turingv2.configuration_turingv2 import TuringV2Config
from marlin.models.turingv2.modeling_turingv2 import TuringV2
config = TuringV2Config()
model = TuringV2Model(config)
```

#### Turing Version 3

```python
from marlin.models.turingv3.configuration_turingv3 import TuringV3Config
from marlin.models.turingv3.modeling_turingv3 import TuringV3
config = TuringV3Config()
model = TuringV3Model(config)
```

## Huggingface compatibility

The classes `MarlinAutoConfig` and `MarlinAutoModel` inherit directly from
the Huggingface classes `transformers.AutoConfig` and `transformers.AutoModel`
respectively. As such they can be used directly to instantiate Huggingface
models. For example `bert-base-uncased` can be loaded as follows:

```python
from marlin.models.auto.configuration_auto import MarlinAutoConfig
from marlin.models.auto.modeling_auto import MarlinAutoModel
config = MarlinAutoConfig.from_pretrained("bert-base-uncased")
model = MarlinAutoModel.from_config(config)
```
