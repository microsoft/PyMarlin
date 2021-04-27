"""
Writers package.
"""
from pymarlin.utils.logger.logging_utils import getlogger
from .base import WriterInitArguments
from .aml import Aml
from .stdout import Stdout
from .tensorboard import Tensorboard
logger = getlogger(__name__)

def build_writer(writer, args: WriterInitArguments):
    """
    Initializes and returns writer object based on writer type.
    """
    logger.debug(f'Building Writer {writer}')
    if writer == 'stdout':
        return Stdout(args)
    if writer == 'aml':
        return Aml()
    if writer == 'tensorboard':
        return Tensorboard(args)
    logger.error(f'Error initializing writer {writer}')
    raise Exception(f"Invalid writer type:{writer} requested.")
