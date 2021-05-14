"""
AML writer module.
"""
from pymarlin.utils.logger.logging_utils import getlogger
from .base import Writer

class Aml(Writer):
    """
    This class implements the Azure ML writer for stats.
    """
    def __init__(self):
        super().__init__(getlogger(__name__))
        self.run = None
        try:
            from azureml.core.run import Run
            self.run = Run.get_context()
            self.logger.info(self.run.get_status())
        except Exception: # pylint: disable=broad-except
            self.run = None
            self.logger.warning('AML writer failed to initialize.')
        self.logger.info(f'run = {self.run}')

    def log_scalar(self, k, v, step):
        """
        Log metric to AML.
        """
        kwargs = {
            'global_step': step,
            k: v
            }
        if self.run is not None:
            self.run.log_row(k, **kwargs)

    def log_multi(self, k, v, step):
        """
        Log metrics to stdout.
        """
        for key, val in v.items():
            key = k+'/'+key
            self.log_scalar(key, val, step)
