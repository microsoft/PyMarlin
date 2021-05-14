"""
Logging util module
"""
import logging
import sys

logging.root.handlers = []
logging.basicConfig(level="WARN",
                    format='SystemLog: %(asctime)s:%(levelname)s : %(name)s : %(lineno)d : %(message)s',
                    stream=sys.stdout)

def getlogger(name, log_level='INFO'):
    """
    This method returns a logger object to be used by the calling class.
    The logger object returned has the following format for all the logs:
    'SystemLog: %(asctime)s:%(levelname)s : %(name)s : %(lineno)d : %(message)s'

    Args:
    name (str): Directory under which to search for checkpointed files.
    file_prefix (str): Prefix to match for when searching for candidate files.
    file_ext (str, optional): File extension to consider when searching.

    Returns:
        logger (object): logger object to use for logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger

if __name__ == '__main__':
    # pylint: disable=pointless-string-statement
    """
    Running this command: "python logging_utils.py" will print following to console:
    SystemLog: logging level for logger1 is INFO
    SystemLog: logging level for logger2 is DEBUG
    SystemLog: <timestamp>:ERROR: logger1 : 34: hello printing error message here for l1
    SystemLog: <timestamp>:ERROR: logger2 : 35: hello printing error message here for l2
    SystemLog: <timestamp>:DEBUG: logger2 : 36: hello printing debug message here for l2
    SystemLog: <timestamp>:INFO: logger2 : 37: hello printing info message here for l2
    """
    l1 = getlogger('logger1')
    l2 = getlogger('logger2', log_level='DEBUG')
    l1.error('hello printing error message here for l1')
    l2.error('hello printing error message here for l2')
    l2.debug('hello printing debug message here for l2')
    l2.info('hello printing info message here for l2')
