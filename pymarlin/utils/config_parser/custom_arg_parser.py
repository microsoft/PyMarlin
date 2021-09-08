"""
Custom Arguments Parser
"""
import argparse
import os
from typing import Dict
import json
import yaml
from pymarlin.utils.logger.logging_utils import getlogger

class CustomArgParser:
    '''
    This class is part of utils and is provided to load arguments from the provided YAML config file.
    Further, the default values of arguments from config file can be overridden via commandline or via
    the special argument --params provided for easy AML experimentation. The class instance takes in the
    parser object and optional log_level. This class needs to be instantiated in the main method inside
    the Marlin_Scenario code.

    Example for instantiation::

        parser = CustomArgParser()
        config = parser.parse()

    The command line arguments to override default YAML config values are passed by adding a '.' between
    namespace and the specific argument as shown in example below. If no namespace is present, then just
    pass the argument name. All command line arguments are optional and need to be prefixed with '--'.
    All commandline arguments not present in YAML config file will be ignored with a warning message.
    Example commandline override::

    python train.py --tmgr.epochs 4 --chkp.save_dir "tmp\\checkpoints"

    NOTE:
    Supported types for CustomArgParser are int, float, str, lists. null is inferred implicitly as str.
    If you intend to use other types, then please set a dummy default value in YAML file and pass the
    intended value from commandline. Suggested defaults::

        str: null
        int: -1
        float: -1.0
        bool: pick either True or False
        list[int]: [-1, -1, -1]
        list[float] : [-1.0, -1.0, -1.0]

    To make it easy to use with AML pipelines, the parser treats --params as a special argument. The parser
    treats this as a str format of JSON serialized dictionary and parses this single argument to override
    the default values from YAML config file of all the arguments present in the dictionary. For example when
    user provides this str for --param::

        '{"--test.test_str":"this is a new test string 2","--test.test_false":"false"}'
        NOTE use of single quote and double quotes and use the same format to avoid parsing errors.

    The parser parses this and overrides the default values provided for test_str and test_false under test
    section in the YAML config file.


    '''
    def __init__(self, yaml_file_arg_key='config_path', default_yamlfile="config.yaml", log_level='INFO'):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--' + yaml_file_arg_key, type=str, default=default_yamlfile, help='Path to YAML config')
        self.parser.add_argument('--params', type=str, default='{}', help='JSON string of string and numeric parameters')
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
        params_dict = None
        for cmd_arg in cmdline_args.keys():
            if cmd_arg == 'params':
                params_dict = json.loads(cmdline_args[cmd_arg])
                continue
            self._parse_arg_and_update_config(cmd_arg, cmdline_args)
        self.logger.debug(f"params_dict is: {params_dict}")
        self._update_from_params_dict(params_dict)
        return self._config

    # config_path as directory, expects a directory with only one .yaml file
    def _resolve_file_from_path(self, file_or_directory: str) -> str:
        if os.path.isdir(file_or_directory):
            yaml_files = [f for f in os.listdir(file_or_directory) if f.endswith('.yaml')]
            if len(yaml_files) == 0:
                raise Exception(f'Could not find any yaml files in directory {file_or_directory}')
            first_yaml_file = yaml_files[0]
            return os.path.join(file_or_directory, first_yaml_file)

        return file_or_directory

    def _parse_config(self, config_path):
        config_path = os.path.abspath(config_path)
        config_path = self._resolve_file_from_path(config_path)
        self.logger.debug(f"absolute config_path = {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as stream:
                try:
                    self._config = yaml.safe_load(stream)
                except Exception as ex:
                    self.logger.error(f"Cannot parse the provided YAML file. Hit this exception: {ex}")
                    raise

        except FileNotFoundError as ex:
            self.logger.error(f"Cannot find provided YAML file. Hit this exception: {ex}")
            raise

    def _update_from_params_dict(self, params_dict):
        if params_dict is None:
            return
        for cmd_arg in params_dict.keys():
            self._parse_arg_and_update_config(cmd_arg, params_dict)

    def _parse_arg_and_update_config(self, arg, arg_dict):
        if '.' in arg:
            arglist = arg.split('.')
            config_iter = self._config[arglist[0].strip('-')]
            idx = 1
            while idx < len(arglist) - 1:
                config_iter = config_iter[arglist[idx]]
                idx = idx + 1
            config_iter[arglist[idx]] = arg_dict[arg]
        else:
            self._config[arg] = arg_dict[arg]

    def _add_arguments(self, cmdline_args):
        for cmd_arg in cmdline_args:
            try:
                if '.' not in cmd_arg:
                    yaml_arg_value = self._config[cmd_arg.strip('-')]
                else:
                    arglist = cmd_arg.split('.')
                    yaml_arg_value = self._config[arglist[0].strip('-')]
                    idx = 1
                    while idx < len(arglist):
                        yaml_arg_value = yaml_arg_value[arglist[idx]]
                        idx = idx + 1
            except Exception as ex: # pylint: disable=broad-except
                self.logger.warning(f"cmd_line arg {cmd_arg} not found in YAML file. Ignoring ex:{ex}")
                continue
            self._add_known_arguments_to_parser(cmd_arg, yaml_arg_value)

    def _add_known_arguments_to_parser(self, arg, value):
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
