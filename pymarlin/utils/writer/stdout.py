"""
Stdout writer module.
"""
from pymarlin.utils.logger.logging_utils import getlogger
from .base import Writer, WriterInitArguments

class Stdout(Writer):
    """
    This class implements the stdout writer for stats.
    """
    def __init__(self, args: WriterInitArguments):
        super().__init__(getlogger(__name__))
        self.args = args

    def log_scalar(self, k, v, step):
        """
        Log metric to stdout.
        """
        self.logger.info(f'step = {step}, {k} : {v}')

    def log_multi(self, k, v, step):
        """
        Log metric to stdout.
        """
        self.logger.info(f'step = {step}, {k} : {v}')

    def log_model(self, flat_weights, flat_grads, step):
        """
        Log model to stdout.
        Can slow down training. Only use for debugging.
        It's logged in Tensorboard by default.
        """
        if self.args.model_log_level == 'DEBUG':
            for name in flat_weights:
                weight_norm = flat_weights[name].norm().item()
                grad_norm = None
                if name in flat_grads:
                    grad_norm = flat_grads[name].norm().item()
                self._log_norms(step, name, weight_norm, grad_norm)

    def log_graph(self, model, device=None):
        """
        Log model graph to stdout.
        """
        self.logger.debug('Logging model graph')
        self.log_multi_line(str(model))

    def _log_norms(self, step, param_name, weight_norm, grad_norm):
        self.logger.debug(f'step = {step} , {param_name} : weight_norm = {weight_norm}, grad_norm = {grad_norm}')
