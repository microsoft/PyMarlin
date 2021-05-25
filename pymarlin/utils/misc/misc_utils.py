"""
Miscellaneous utility functions
"""

from functools import wraps
import os
import re
import shutil
from pymarlin.utils.logger.logging_utils import getlogger

#https://docs.python.org/2/library/functools.html#functools.wraps

def snake2camel(name):
    """
    This method changes input name from snake format to camel format.

    Args:
        name (str): snake format input name.

    Returns:
        name (str): camel format input name.

    """
    return re.sub(r'(?:^|_)([a-z])', lambda x: x.group(1).upper(), name)

def clear_dir(path, skips=None):
    """
    This method deletes the contents of the directory for which path
    has been provided and not included in the skips list.

    Args:
        path (str): Path for directory to be deleted.
        skips (List[str]): List of paths for sub directories to be skipped from deleting.

    """
    if os.path.isdir(path):
        with os.scandir(path) as path_iter:
            for entry in path_iter:
                if entry.path in skips:
                    continue
                try:
                    if entry.is_file() or entry.is_symlink():
                        os.remove(entry.path)
                    else:
                        shutil.rmtree(entry.path)
                except PermissionError:
                    getlogger(__name__).warning(f"could not delete path: {entry.path}")

def debug(method):
    """
    This method wraps input method with debug calls to measure time taken for
    the given input method to finish.

    Args:
        method (function): Method which needs to be timed.

    Returns:
        debugged (method): debugged function.

    """
    @wraps(method)
    def debugged(*args, **kw):
        logger = getlogger(__name__)
        logger.debug('Inside method: %s', method.__name__)
        result = method(*args, **kw)
        logger.debug('Finished method: %s', method.__name__)
        return result
    return debugged
