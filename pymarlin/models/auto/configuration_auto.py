import os
from collections import OrderedDict
import json

from transformers import AutoConfig

from ..turingv1.configuration_turingv1 import TuringV1Config
from ..turingv2.configuration_turingv2 import TuringV2Config
from ..turingv3.configuration_turingv3 import TuringV3Config

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

CONFIG_MAPPING = OrderedDict(
    [
        ("turingv1", TuringV1Config),
        ("turingv2", TuringV2Config),
        ("turingv3", TuringV3Config),
    ]
)

class MarlinAutoConfig(AutoConfig):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Extend Huggingface AutoConfig to Turing family.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Specify the configuration class to be instantiated.

        Example::
            
            >>> from pymarlin.models.auto.configuration_auto import MarlinAutoConfig
            >>> from pymarlin.models.auto.modeling_auto import MarlinAutoModel
            >>> config = MarlinAutoConfig.from_pretrained("turingv1-base-uncased")
            >>> model = MarlinAutoModel.from_config(config)
        """
        if pretrained_model_name_or_path in MARLIN_PRETRAINED_MODEL_NAMES:
            config_path = cls._get_turing_config_dict(pretrained_model_name_or_path)

            with open(config_path) as f:
                config_dict = json.load(f)

            if "model_type" in config_dict:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
                return config_class.from_dict(config_dict, **kwargs)
            else:
                # pattern match on string as in huggingface implementation
                for pattern, config_class in CONFIG_MAPPING.items():
                    if pattern in str(pretrained_model_name_or_path):
                        return config_class.from_dict(config_dict, **kwargs)

        else:
            return super(MarlinAutoConfig, cls).from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

    @staticmethod
    def _get_turing_config_dict(pretrained_model_name_or_path):
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.abspath(os.path.join(dirname, os.pardir, "configs"))
        if "base" in pretrained_model_name_or_path:
            config_path = os.path.join(config_dir, "base-uncased.json")
        elif "large" in pretrained_model_name_or_path:
            config_path = os.path.join(config_dir, "large-uncased.json")
        elif "mini" in pretrained_model_name_or_path:
            config_path = os.path.join(config_dir, "mini-uncased.json")
        else:
            raise ValueError(
                f"Unknown pretraining_model_name_or_path {pretrained_model_name_or_path}, currently support for Turing models is base (12), large (24) or mini (2)"
            )
        return config_path
