"""
Tensorboard writer module.
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from pymarlin.utils.logger.logging_utils import getlogger
from pymarlin.utils.misc.misc_utils import clear_dir
from .base import Writer, WriterInitArguments

# Workaround for standard image which includes both tf and tb
# More details here: https://github.com/pytorch/pytorch/issues/30966
try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except ImportError:
    pass

class Tensorboard(Writer):
    """
    This class implements a wrapper on Tensorboard's SummaryWriter
    for logging stats to Tensorboard. Please look here for detailed information
    on each function:  https://pytorch.org/docs/stable/tensorboard.html#
    Visit this blog for more examples on logging stats to Tensorboard:
    https://krishansubudhi.github.io/deeplearning/2020/03/24/tensorboard-pytorch.html
    """
    def __init__(self, args: WriterInitArguments):
        super().__init__(getlogger(__name__))
        self.args = args

        log_dir = self.args.tb_log_dir
        if self.args.tb_logpath_parent_env and self.args.tb_logpath_parent_env in os.environ:
            parent_dir = os.getenv(self.args.tb_logpath_parent_env)
            log_dir = os.path.join(parent_dir, self.args.tb_log_dir)

        azureml_dirs = ['logs/azureml']
        clear_dir(log_dir, skips=azureml_dirs)
        self.logger.info(f'Cleared directory {log_dir} (skipping azureml dirs)')

        os.makedirs(log_dir, exist_ok=True)
        self.logger.info(f'Created tensorboard folder {log_dir} : {os.listdir(log_dir)}')
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, k, v, step):
        """
        Log metric to Tensorboard graph.
        """
        self.writer.add_scalar(k, v, step)

    def log_multi(self, k, v, step):
        """
        Log multiple metrics in the same Tensorboard graph.
        """
        if self.args.tb_log_multi:
            self.writer.add_scalars(k, v, step)
        else:
            for key, val in v.items():
                key = k+'/'+key
                self.writer.add_scalar(key, val, step)

    def log_model(self, flat_weights, flat_grads, step):
        """
        Log model weights and gradients to Tensorboard.
        """
        weight_norms = {}
        grad_norms = {}
        for name in flat_weights:
            self.log_histogram('weights/'+name, flat_weights[name], step)
            weight_norm = flat_weights[name].norm().item()
            weight_norms[name] = weight_norm

            grad_norm = None
            if name in flat_grads:
                self.log_histogram('grads/'+name, flat_grads[name], step)
                grad_norm = flat_grads[name].norm().item()
                grad_norms[name] = grad_norm

            self._log_norms(step, weight_norms, grad_norms)

    def log_embedding(self, tag, mat, labels, step):
        """
        Log model embeddings to Tensorboard.
        """
        self.writer.add_embedding(mat, tag=tag, metadata=labels, global_step=step)

    def log_graph(self, model, device):
        """
        Logs model graphs to Tensorboard.

        Args:
        model (object): unwrapped model with a function get_sample_input() implemented.
        device (str): device type.
        """
        if hasattr(model, 'get_sample_input'):
            inputs = model.get_sample_input()
            inputs_to_device = ()
            if isinstance(inputs, tuple):
                for item in inputs:
                    inputs_to_device += (item.to(device),)
            else:
                inputs_to_device = inputs.to(device)
            self.writer.add_graph(model, inputs_to_device)
        else:
            self.logger.warning('Could not log model graph. Implement get_sample_input() in model class')

    def log_image(self, k, v, step, dataformats='HW'):
        """
        Log image in Tensorboard.
        """
        self.writer.add_image(k, v, step, dataformats=dataformats)

    def log_pr_curve(self, k, preds, labels, step):
        """
        Log Precision Recall curve in Tensorboard.
        """
        self.writer.add_pr_curve(k, labels=labels, predictions=preds, global_step=step, weights=None)

    def log_args(self, args):
        """
        Log all the Arguments used in the experiment to Tensorboard.
        """
        self.logger.info('Logging Arguments for this experiment to Tensorboard.')
        smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='elr_tb_args'))
        writer = self.writer._get_file_writer() # pylint: disable=protected-access
        for key, value in args.items():
            self.logger.debug(f'key = {key}, value = {value}')
            tensor = TensorProto(dtype='DT_STRING',
                                 string_val=[str(value).encode(encoding='utf_8')],
                                 tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
            summary = Summary(value=[Summary.Value(tag=key, tensor=tensor, metadata=smd)])
            writer.add_summary(summary=summary)

    def log_histogram(self, param_name, vals, step):
        """
        Log histograms in Tensorboard.
        Avoid using small step size since it impact training time.
        """
        if step % self.args.tb_log_hist_steps == 0:
            self.logger.info(f'Logging histogram for step {step}')
            if torch.isfinite(vals).all().item():
                self.writer.add_histogram(param_name, vals, step)
            else:
                self.logger.warning('nan found while logging histogram')

    def _log_norms(self, step, weight_norms, grad_norms):
        """
        Logs weight and grad norms.

        Args:
        weight_norms (List[str]): norms of weights of all layers of model that needs to be logged.
        grad_norms (List[str]): norms of grads of all layers of model that needs to be logged.
        """

        for name in weight_norms:
            self.log_scalar('weight_norm/'+name, weight_norms[name], step)
        for name in grad_norms:
            self.log_scalar('grad_norm/'+name, grad_norms[name], step)

    def flush(self):
        """
        Flush the SummaryWriter to write out Summary to Tensorboard.
        """
        self.writer.flush()

    def finish(self):
        """
        Flush the SummaryWriter to write out Summary to Tensorboard and
        close SummaryWriter.
        """
        self.writer.flush()
        self.writer.close()
