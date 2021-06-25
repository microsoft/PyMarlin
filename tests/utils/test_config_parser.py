'''
Test for config parser module.
'''
from dataclasses import dataclass, field
from typing import List, Optional
from unittest import mock, TestCase
from pymarlin.utils.config_parser.custom_arg_parser import CustomArgParser


@dataclass
class Args:
    test_float: float = -1.0
    test_int: int = -1
    test_list_float: List = field(default_factory=lambda: [-1.0])
    test_list_int: List = field(default_factory=lambda: [-1])
    test_list_str: List = field(default_factory=lambda: [None])
    test_true: bool = False
    test_false: bool = True
    test_str: str = None
    optional_arg: Optional[int] = None

class TestConfigParser(TestCase):
    def test_config_parser(self):
        cmdline_args = ["test", "--config_path", ".//tests//utils//config.yaml", "--test.test_true", "t", "--test.test_false", "false", "--dummy", "--test.test_str", "this is a test str."]
        # cmdline_args = ["test", "--config_path", ".//config.yaml", "--test.test_true", "t", "--test.test_false", "false", "--dummy", "--test.test_str", "this is a test str."]
        with mock.patch('sys.argv', cmdline_args):      
            parser = CustomArgParser(log_level='DEBUG')
            config = parser.parse()
            self.assertTrue(config['test'] is not None)
            args = Args(**config['test'])
            self.assertTrue(args.test_true)
            self.assertFalse(args.test_false)
            self.assertTrue(args.test_list_float == [-1.0, -1.0, -1.0])
            self.assertTrue(args.test_str == "this is a test str.")
            self.assertTrue(args.test_list_str == [ 'this', 'is', 'a', 'test', 'list'])
    
    def test_config_parser_invalid_boolean(self):
        cmdline_args = ["test", "--config_path", ".//tests//utils//config.yaml", "--test.test_true", "fl", "--dummy", "--test.test_str", "this is a test str."]
        # cmdline_args = ["test", "--config_path", ".//config.yaml", "--test.test_true", "fl", "--dummy", "--test.test_str", "this is a test str."]
        with mock.patch('sys.argv', cmdline_args):
            parser = CustomArgParser(log_level='DEBUG')
            self.assertRaises(SystemExit, parser.parse)
    
    def test_parse_config_incorrect_path(self):
        parser = CustomArgParser(log_level='DEBUG')
        self.assertRaises(FileNotFoundError, parser._parse_config, ".//dummy_path")
    
    def test_parse_config_corrupt_file(self):
        cmdline_args = ["test", "--config_path", ".//tests//utils//config.yaml"]
        # cmdline_args = ["test", "--config_path", ".//config.yaml"]
        with mock.patch('sys.argv', cmdline_args):
            parser = CustomArgParser(log_level='DEBUG')
            self.assertRaises(Exception, parser._parse_config, ".//corrupt_files//config.yaml")
    
    def test_add_arguments_invalid(self):
        cmdline_args = ["test", "--config_path", ".//tests//utils//config.yaml"]
        # cmdline_args = ["test", "--config_path", ".//config.yaml"]
        with mock.patch('sys.argv', cmdline_args):
            parser = CustomArgParser(log_level='DEBUG')
            parser._add_arguments(['--dummy'])
    
    def test_params(self):
        cmdline_args = ["test", "--config_path", ".//tests//utils//config.yaml", "--params", '{"test.test_str": "hello world!"}']
        # cmdline_args = ["test", "--config_path", ".//config.yaml"]
        with mock.patch('sys.argv', cmdline_args):
            parser = CustomArgParser(log_level='DEBUG')
            config = parser.parse()
            self.assertTrue(config['test'] is not None)
            args = Args(**config['test'])
            self.assertTrue(args.test_str == 'hello world!')
    
    def test_dir_configpath(self):
        cmdline_args = ["test", "--config_path", ".//tests//utils//"]
        with mock.patch('sys.argv', cmdline_args):
            parser = CustomArgParser(log_level='DEBUG')
            config = parser.parse()
            self.assertTrue(config['test'] is not None)
            args = Args(**config['test'])
            self.assertTrue(args.test_list_int == [-1, -1, -1])
            self.assertTrue(args.test_list_str == [ 'this', 'is', 'a', 'test', 'list'])