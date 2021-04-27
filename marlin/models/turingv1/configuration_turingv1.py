from ..turing_utils.configuration_turing import TuringConfig

class TuringV1Config(TuringConfig):
    """Configuration class for Turing NLG V1 model.

    Example:
        >>> from marlin.models.turingv1.configuration_turingv1 import TuringV1Config
        >>> from marlin.models.turingv1.modeling_turingv1 import TuringV1Model
        >>> config = TuringV1Config()
        >>> model = TuringV1Model(config)
    """
    
    model_type = "turingv1"
