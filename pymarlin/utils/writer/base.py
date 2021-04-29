"""
Base class for Writers
"""
from abc import ABC
import dataclasses

@dataclasses.dataclass
class WriterInitArguments:
    """
    Writer Arguments.
    """
    tb_log_dir: str = 'logs'
    tb_logpath_parent_env: str = None
    tb_log_multi: bool = False
    tb_log_hist_steps: int = 20000
    model_log_level: str = 'INFO'

class Writer(ABC):
    """
    Abstract Base class for Writers.
    """
    def __init__(self, logger):
        self.logger = logger

    def log_scalar(self, k, v, step):
        pass

    def log_multi(self, k, v, step):
        pass

    def log_model(self, flat_weights, flat_grads, step):
        pass

    def log_args(self, args):
        pass

    def log_graph(self, model, device=None):
        pass

    def log_image(self, k, v, step, dataformats='HW'):
        pass

    def log_pr_curve(self, k, preds, labels, step):
        pass

    def log_histogram(self, param_name, vals, step):
        pass

    def log_embedding(self, tag, mat, labels, step):
        pass

    def _log_norms(self, step, param_name, weight_norm, grad_norm):
        pass

    def log_multi_line(self, string):
        lines = string.split('\n')
        for line in lines:
            self.logger.info(line)

    def flush(self):
        pass

    def finish(self):
        pass
