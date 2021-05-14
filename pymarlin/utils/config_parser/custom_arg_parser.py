"""
Custom Arguments Parser
"""
import argparse
import os
from typing import Dict
import yaml
from pymarlin.utils.logger.logging_utils import getlogger

class CustomArgParser:
    r"""
    This class is part of utils and is provided to load arguments from the provided YAML config file.
    Further, the default values of arguments from config file can be overridden via command line.
    The class instance takes in the parser object and optional log_level.
    This class needed to be instantiated in the main method inside the ELR_Scenario code.

    ''Example for instantiation'':
        parser = CustomArgParser()
        config = parser.parse()

    The command line arguments to override default YAML config values are passed by adding a '.' between
    namespace and the specific argument as shown in example below. If no namespace is present, then just
    pass the argument name. All command line arguments are optional and need to be prefixed with '--'.
    All commandline arguments not present in YAML config file will be ignored with a warning message.
    Example commandline override:
    python train.py --tmgr.epochs 4 --chkp.save_dir "tmp\checkpoints"

    NOTE:
    Supported types for CustomArgParser are int, float, str, lists. null is inferred implicitly as str.
    If you intend to use other types, then please set a dummy default value in YAML file and pass the
    intended value from commandline. Suggested defaults:
        str: null
        int: -1
        float: -1.0
        bool: pick either True or False
        list[int]: [-1, -1, -1]
        list[float] : [-1.0, -1.0, -1.0]

    """
    def __init__(self, yaml_file_arg_key='config_path', default_yamlfile="config.yaml", log_level='INFO'):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--' + yaml_file_arg_key, type=str, default=default_yamlfile, help='Path to YAML config')
        self._config = None
        self.logger = getlogger(__name__, log_level)

    def parse(self) -> Dict:
        """
        Parse YAML config file, parse commandline arguments and merge the two
        to get the final merged config dictionary.

        Find and return the path of the file with greatest number of
        completed epochs under dirpath (recursive search) for a given file
        prefix, and optionally file extension.

        Args:
            self

        Returns:
            self._config (Dict): merged config dictionary containing all arguments.
        """
        args_raw, cmdline_args_raw = self.parser.parse_known_args()
        self.logger.debug(f"args_raw are : {args_raw}, cmdline_args_raw are: {cmdline_args_raw}")
        cmdline_args_list = [cmd_arg for cmd_arg in cmdline_args_raw if cmd_arg.startswith('--')]
        args, _ = self.parser.parse_known_args()
        self._parse_config(args.config_path)

        self._add_arguments(cmdline_args_list)
        args, unknown_args = self.parser.parse_known_args()
        self.logger.debug(f"parsed args are : {args}, unknown args are: {unknown_args}")
        cmdline_args = vars(args)
        for cmd_arg in cmdline_args.keys():
            if '.' in cmd_arg:
                arglist = cmd_arg.split('.')
                self._config[arglist[0]][arglist[1]] = cmdline_args[cmd_arg]
            else:
                self._config[cmd_arg] = cmdline_args[cmd_arg]
        return self._config

    def _parse_config(self, config_path):
        config_path = os.path.abspath(config_path)
        self.logger.debug(f"absolute config_path = {config_path}")
        try:
            with open(config_path) as stream:
                try:
                    self._config = yaml.safe_load(stream)
                except Exception as ex:
                    self.logger.error(f"Cannot parse the provided YAML file. Hit this exception: {ex}")
                    raise

        except FileNotFoundError as ex:
            self.logger.error(f"Cannot find provided YAML file. Hit this exception: {ex}")
            raise

    def _add_arguments(self, cmdline_args):
        for cmd_arg in cmdline_args:
            try:
                if '.' not in cmd_arg:
                    yaml_arg_value = self._config[cmd_arg.strip('-')]
                else:
                    arglist = cmd_arg.split('.')
                    yaml_arg_value = self._config[arglist[0].strip('-')][arglist[1]]
            except Exception as ex: # pylint: disable=broad-except
                self.logger.warning(f"cmd_line arg {cmd_arg} not found in YAML file. Ignoring ex:{ex}")
                continue
            self._add_known_arguments_to_parser(cmd_arg, yaml_arg_value)

    def _add_known_arguments_to_parser(self, arg, value):
        # TODO: Add support for dictionary parsing
        if value is None:
            self.parser.add_argument(arg, type=str, default=None)
        elif isinstance(value, bool):
            self.parser.add_argument(arg, type=self._str2bool, nargs='?', const=True, default=value)
        elif isinstance(value, list):
            if isinstance(value[0], (int, str, float)):
                self.parser.add_argument(arg, \
                    type=lambda uf: self._eval_str_list(uf, eval_type=type(value[0])))
            else:
                self.logger.warning(f"unsupported type(yaml_arg_value): {type(value)}")
        else:
            self.parser.add_argument(arg, type=type(value), default=value)

    def _eval_str_list(self, x, eval_type=float):
        if x is None:
            return None
        if isinstance(x, str):
            if '-' in x:
                x = x.split('-')
            else:
                x = eval(x) # pylint: disable=eval-used
        try:
            return list(map(eval_type, x))
        except TypeError:
            return [eval_type(x)]

    def _str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
