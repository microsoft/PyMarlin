from ..turing_utils.configuration_turing import TuringConfig

class TuringV2Config(TuringConfig):
    """Configuration class for Turing NLG V1 model.

    Example:
        >>> from marlin.models.turingv2.configuration_turingv2 import TuringV2Config
        >>> from marlin.models.turingv2.modeling_turingv2 import TuringV2Model
        >>> config = TuringV2Config()
        >>> model = TuringV2Model(config)
    """
    
    model_type = "turingv2"
