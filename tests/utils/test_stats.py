"""Module to test stats module class"""
import os
import torch
import numpy as np
import pytest
import shutil
import unittest
from unittest import mock
from marlin.utils import stats
from marlin.utils.writer import build_writer, WriterInitArguments
import collections
import functools

class TestStats(unittest.TestCase):
    def setUp(self):
        self.stats = stats.global_stats
        self.stat_args = stats.StatInitArguments(
                log_steps = 50,
                update_system_stats = False,
                log_model_steps = 1000,
                exclude_list = None
        )
        self.writer_args = WriterInitArguments(
                tb_log_dir='logs'
        )
        self.writers = [
                build_writer(writer, self.writer_args)
                if isinstance(writer, str)
                else writer
                for writer in ['stdout','tensorboard']
        ]
        self.stats.rebuild(args=self.stat_args, writers=self.writers)

    def tearDown(self):
        self.stats.rebuild(args=None, writers=[])

    @pytest.fixture(scope='module')
    def project_file(self, tmpdir_factory):
        print('deleting temp folder')
        my_tmpdir = tmpdir_factory.mktemp(self.writer_args.tb_log_dir)
        yield my_tmpdir
        shutil.rmtree(str(my_tmpdir))

    def test_short(self):
        scalars = {'F1': 0.5, 'acc':0.8}
        for k, v in scalars.items():
            self.stats.update(k,v, frequent = True)
            assert self.stats.scalars_short[k] == v
            self.stats.update(k,v+0.1, frequent = True)
            assert self.stats.scalars_short[k] == v+0.1
        self.stat_args.log_steps = 2
        self.stats.rebuild(args=self.stat_args, writers=self.writers)
        print('log stats for step 1. Nothing should be logged here.')
        self.stats.log_stats(step = 1)
        assert len(self.stats.scalars_short) > 0
        print('log stats for step 2. should be logged now.')
        self.stats.log_stats(step = 2)
        assert len(self.stats.scalars_short) == 0

    def test_long(self):
        scalars = {'epochs': 1}
        for k,v in scalars.items():
            self.stats.update(k,v, frequent = False)
            assert self.stats.scalars_long[k] == v
        multi = {'losses': {'train':0.5, 'val_email':0.8, 'val_wiki':0.3}}
        for k,v in multi.items():
            self.stats.update_multi(k,v, frequent = False)
            assert self.stats.multi_long[k] == v
        print('log long stats . should be logged')
        self.stats.log_long_stats(step = 1000)
    
    def test_log_model(self):
        # 2 layer NN with layer norm and sigmoid
        model = MyModel()
        self.stats.log_graph(model, device='cpu')
        optim = torch.optim.SGD(params = model.parameters(), lr = 1)
        
        self.stat_args.log_steps = 1
        self.writer_args.tb_hist_interval = 2
        self.writers = [
                build_writer(writer, self.writer_args)
                if isinstance(writer, str)
                else writer
                for writer in ['stdout','tensorboard']
        ]
        self.stats.rebuild(args=self.stat_args, writers=self.writers)
        
        for step in range(1, 5):
            op = model.forward(torch.rand(2,3))
            loss = torch.nn.MSELoss()(op, torch.rand(2,1))
            loss.backward()
            
            self.stats.log_model(step, model)
            optim.step()
            optim.zero_grad()
        #expectation. norms should be logged 4 times
        #histogram should be logged only twice in tensorboard

    def test_log_image(self):
        random_image = np.random.randint(100,size = (1,100)).reshape(10,10)
        random_image = random_image/ 100
        self.stats.update_image('random_image',
                        random_image,
                        dataformats = 'HW')
        self.stats.log_long_stats(step = 1000)

    def test_log_pr(self):
        preds = np.random.rand(100)
        labels = np.random.randint(2, size=100)
        self.stats.update_pr('binary_pr',
                        preds, labels)
        self.stats.log_long_stats(step = 1000)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(3,5)
        self.hidden_activation = torch.nn.Tanh()
        self.hidden_layernorm = torch.nn.LayerNorm(5)
        self.output = torch.nn.Linear(5,1)
    
    def forward(self, input):
        hidden_op = self.hidden_activation(
                        self.hidden_layernorm(
                            self.hidden(input)))
        op = self.output(hidden_op)
        return op
    
    def get_sample_input(self):
        return torch.ones(1,3, dtype = torch.float32)
