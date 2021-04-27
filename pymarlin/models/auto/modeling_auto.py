import os
from collections import OrderedDict

import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

# turing v1 imports
from ..turingv1.configuration_turingv1 import TuringV1Config
from ..turingv1.modeling_turingv1 import (
    TuringV1Model,
    TuringV1ForSequenceClassification,
    TuringV1ForTokenClassification,
)

# turing v2 imports
from ..turingv2.configuration_turingv2 import TuringV2Config
from ..turingv2.modeling_turingv2 import (
    TuringV2Model,
    TuringV2ForSequenceClassification,
    TuringV2ForTokenClassification,
)

# turing v1 imports
from ..turingv3.configuration_turingv3 import TuringV3Config
from ..turingv3.modeling_turingv3 import (
    TuringV3Model,
    TuringV3ForSequenceClassification,
    TuringV3ForTokenClassification,
)

MARLIN_PRETRAINED_MODEL_NAMES = set(
    [
        "turingv1-mini-uncased",
        "turingv1-base-uncased",
        "turingv1-large-uncased",
        "turingv2-mini-uncased",
        "turingv2-base-uncased",
        "turingv2-large-uncased",
        "turingv3-mini-uncased",
        "turingv3-base-uncased",
        "turingv3-large-uncased",
    ]
)

MARLIN_MODEL_MAPPING = OrderedDict(
    [
        (TuringV1Config, TuringV1Model),
        (TuringV2Config, TuringV2Model),
        (TuringV3Config, TuringV3Model),
    ]
)

MARLIN_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (TuringV1Config, TuringV1ForSequenceClassification),
        (TuringV2Config, TuringV2ForSequenceClassification),
        (TuringV3Config, TuringV3ForSequenceClassification),
    ]
)

MARLIN_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (TuringV1Config, TuringV1ForTokenClassification),
        (TuringV2Config, TuringV2ForTokenClassification),
        (TuringV3Config, TuringV3ForTokenClassification),
    ]
)

class MarlinAutoModel(AutoModel):
    @classmethod
    def from_config(cls, config):
        """
        Extend Huggingface AutoModel to Turing family.

        Args:
            config (:class:`~transformers.PreTrainedConfig`):
                The model class to instantiate is selected based on the configuration class.

        Example::
            
            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModel
            >>> config = MarlinAutoConfig.from_pretrained("turingv1-base-uncased")
            >>> model = MarlinAutoModel.from_config(config)
        """
        if type(config) in MARLIN_MODEL_MAPPING.keys():
            return MARLIN_MODEL_MAPPING[type(config)](config)
        else:
            return super(MarlinAutoModel, cls).from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Example:

            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModel
            >>> config = MarlinAutoConfig.from_pretrained("turingv3-large-uncased")
            >>> model = MarlinAutoModel.from_pretrained("./turingv3-large-uncased.pt", config=config)
        """
        config = kwargs.pop("config", None)

        if type(config) in MARLIN_MODEL_MAPPING:
            # random weights
            model = cls.from_config(config)
            # load pretrained weights
            state_dict = cls._load_state_dict(pretrained_model_name_or_path)
            # update model
            model.load_state_dict(state_dict)
            return model

        else:
            print(f"Unable to find configuration class in {MARLIN_MODEL_MAPPING.keys()}")
            print("Defaulting to Huggingface AutoModel.from_pretrained method.")
            model = super(MarlinAutoModel, cls).from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
            return model
    
    @staticmethod
    def _load_state_dict(state_dict_path):
        try:
            state_dict = torch.load(state_dict_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "File not found. Note - {cls.__name__} only supports providing path to pretrained\n"
                "state_dict in 'pretrained_model_name_or_path'."
            )
        return state_dict

class MarlinAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    @classmethod
    def from_config(cls, config):
        """
        Extend Huggingface AutoModelForSequenceClassification to Turing family.

        Args:
            config (:class:`~transformers.PreTrainedConfig`):
                The model class to instantiate is selected based on the configuration class.

        Example::
            
            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModelForSequenceClassification
            >>> config = MarlinAutoConfig.from_pretrained("turingv1-base-uncased")
            >>> model = MarlinAutoModelForSequenceClassification.from_config(config)
        """
        if type(config) in MARLIN_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys():
            return MARLIN_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[type(config)](config)
        else:
            return super(MarlinAutoModelForSequenceClassification, cls).from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Example:

            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModelForSequenceClassification
            >>> config = MarlinAutoConfig.from_pretrained("turingv3-large-uncased")
            >>> model = MarlinAutoModelForSequenceClassification.from_pretrained("./turingv3-large-uncased.pt", config=config)
        """
        config = kwargs.pop("config", None)

        if type(config) in MARLIN_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING:
            # random weights
            model = cls.from_config(config)
            # load pretrained weights
            state_dict = cls._load_state_dict(pretrained_model_name_or_path)
            # update model
            model.load_state_dict(state_dict)
            return model

        else:
            print(f"Unable to find configuration class in {MARLIN_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()}")
            print("Defaulting to Huggingface AutoModelForSequenceClassification.from_pretrained method.")
            model = super(MarlinAutoModelForSequenceClassification, cls).from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
            return model
    
    @staticmethod
    def _load_state_dict(state_dict_path):
        try:
            state_dict = torch.load(state_dict_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "File not found. Note - {cls.__name__} only supports providing path to pretrained\n"
                "state_dict in 'pretrained_model_name_or_path'."
            )
        return state_dict

class MarlinAutoModelForTokenClassification(AutoModelForTokenClassification):
    @classmethod
    def from_config(cls, config):
        """
        Extend Huggingface AutoModelForTokenClassification to Turing family.

        Args:
            config (:class:`~transformers.PreTrainedConfig`):
                The model class to instantiate is selected based on the configuration class.

        Example::
            
            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModelForTokenClassification
            >>> config = MarlinAutoConfig.from_pretrained("turingv1-base-uncased")
            >>> model = MarlinAutoModelForTokenClassification.from_config(config)
        """
        if type(config) in MARLIN_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys():
            return MARLIN_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[type(config)](config)
        else:
            return super(MarlinAutoModelForTokenClassification, cls).from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Example:

            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModelForTokenClassification
            >>> config = MarlinAutoConfig.from_pretrained("turingv3-large-uncased")
            >>> model = MarlinAutoModelForTokenClassification.from_pretrained("./turingv3-large-uncased.pt", config=config)
        """
        config = kwargs.pop("config", None)

        if type(config) in MARLIN_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING:
            # random weights
            model = cls.from_config(config)
            # load pretrained weights
            state_dict = cls._load_state_dict(pretrained_model_name_or_path)
            # update model
            model.load_state_dict(state_dict)
            return model

        else:
            print(f"Unable to find configuration class in {MARLIN_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()}")
            print("Defaulting to Huggingface AutoModelForTokenClassification.from_pretrained method.")
            model = super(MarlinAutoModelForTokenClassification, cls).from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
            return model
    
    @staticmethod
    def _load_state_dict(state_dict_path):
        try:
            state_dict = torch.load(state_dict_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "File not found. Note - {cls.__name__} only supports providing path to pretrained\n"
                "state_dict in 'pretrained_model_name_or_path'."
            )
        return state_dict