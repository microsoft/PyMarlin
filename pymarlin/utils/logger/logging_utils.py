"""
Logging util module
"""
import logging

# create console handler for pymarlin format
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s:%(levelname)s : %(name)s : %(lineno)d : %(message)s')
console_handler.setFormatter(formatter)

def getlogger(name, log_level='INFO'):
    """
    This method returns a logger object to be used by the calling class.
    The logger object returned has the following format for all the logs:
    '%(asctime)s:%(levelname)s : %(name)s : %(lineno)d : %(message)s'

    Args:
    name (str): Directory under which to search for checkpointed files.
    file_prefix (str): Prefix to match for when searching for candidate files.
    file_ext (str, optional): File extension to consider when searching.

    Returns:
        logger (object): logger object to use for logging.
    """
    logger = logging.getLogger(name)
    logger.handlers = [console_handler]
    logger.setLevel(log_level)
    return logger

if __name__ == '__main__':
    # pylint: disable=pointless-string-statement
    """
    Running this command: "python logging_utils.py" will print following to console:
    logging level for logger1 is INFO
    logging level for logger2 is DEBUG
    <timestamp>:ERROR: logger1 : 34: hello printing error message here for l1
    <timestamp>:ERROR: logger2 : 35: hello printing error message here for l2
    <timestamp>:DEBUG: logger2 : 36: hello printing debug message here for l2
    <timestamp>:INFO: logger2 : 37: hello printing info message here for l2
    """
    l1 = getlogger('logger1')
    l2 = getlogger('logger2', log_level='DEBUG')
    l1.error('hello printing error message here for l1')
    l2.error('hello printing error message here for l2')
    l2.debug('hello printing debug message here for l2')
    l2.info('hello printing info message here for l2')
