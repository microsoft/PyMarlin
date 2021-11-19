from .trainer_backend import *
from .ort_trainer_backend import ORTTrainerBackend

def build_trainer_backend(trainer_backend_name, *args, **kwargs):
    """Factory for trainer_backends

    Args:
        trainer_backend_name (str): TrainerBackend Name. Possible choices are currently: sp, sp-amp, sp-amp-apex, ddp, ddp-amp, ddp-amp-apex
        args (sequence): TrainerBackend positional arguments
        kwargs (dict): TrainerBackend keyword arguments
    """
    factory_dict = {
        "sp": SingleProcess,
        "sp-amp": SingleProcessAmp,
        "sp-amp-apex": SingleProcessApexAmp,
        "ddp": DDPTrainerBackendFactory(SingleProcess),
        "ddp-amp-ort": DDPORTTrainerBackendFactory(SingleProcessAmp),
        "ddp-amp": DDPTrainerBackendFactory(SingleProcessAmp),
        "ddp-amp-apex": DDPTrainerBackendFactory(SingleProcessApexAmp),
    }
    return factory_dict[trainer_backend_name](*args, **kwargs)

def DDPTrainerBackendFactory(trainer_backend_cls): # pylint: disable=invalid-name
    def create(*args, gather_frequency: Optional[int] = None, **kwargs):
        # pull out args to DDPTrainerBackend if needed here.
        return DDPTrainerBackend(trainer_backend_cls(*args, **kwargs), gather_frequency=gather_frequency)

    return create

# testing TODO: refactor factory logic to do hierachael decoration (sp->ort->ddp/deepspeed)
def DDPORTTrainerBackendFactory(trainer_backend_cls):
    def create(*args, gather_frequency: Optional[int] = None, **kwargs):
        return DDPTrainerBackend(
            ORTTrainerBackend(trainer_backend_cls(*args, **kwargs)), 
            gather_frequency=gather_frequency)

    return create
