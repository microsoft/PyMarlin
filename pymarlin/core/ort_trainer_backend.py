from .trainer_backend import *
import sys
from pymarlin.utils.logger import getlogger
import torch.nn as nn

class ORTTrainerBackend(AbstractTrainerBackendDecorator):   
    def __init__(self, trainer_backend):
        super().__init__(trainer_backend)
        self.logger = getlogger(__file__,log_level='DEBUG')
    
    # TODO: add these under TrainerBackendDecoratorPassThrough, which ORT, Opacus can inherit from
    # so that DDP backend can get/set from wrapped SingleProcess*
    def __getattribute__(self, name):
        # self.logger.debug(f'__getattribute__(name={name})')
        if name in ('trainer_backend','init','__init__','logger', '_core_model', 'core_model') :
            return super().__getattribute__(name)
        else:
            return self.trainer_backend.__getattribute__(name)

    def __setattr__(self, name, value):
        # self.logger.debug(f'__setattr_(name={name},value={value})')
        if name in ('trainer_backend','init','__init__','logger', '_core_model', 'core_model') :
            super().__setattr__(name, value)
        else:
            self.trainer_backend.__setattr__(name, value)

    @property
    def core_model(self):
        return self._core_model
    
    @core_model.setter
    def core_model(self, model):
        self._core_model = model 
    
    def init(self, args: TrainerBackendArguments):
        super().init(args)
        try:
            from torch_ort import ORTModule
        except:
            self.logger.error("could not import ORTModule")
            sys.exit(1)
        
        assert(hasattr(self.trainer_backend.model, 'model'), 'self.trainer_backend.model.model does not exist')
        assert(isinstance(self.trainer_backend.model.model, nn.Module), "expected module_inteface.model of type torch.nn.Module")
        
        # get the reference and save it before ORTModule wrap
        self.core_model = self.trainer_backend.model.model
        module = self.trainer_backend.model # TODO: should we change trainer_backend.model to module?
        module.get_core_model = lambda: self.core_model

        self.logger.info("Wrapping trainer_backend.model.model")
        self.trainer_backend.model.model = ORTModule(self.trainer_backend.model.model)