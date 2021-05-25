"""
Basic Stats module.
"""
from collections.abc import Iterable
import dataclasses
import os
import re
import torch
import numpy as np
from pymarlin.utils.logger.logging_utils import getlogger

try:
    import psutil
except ImportError:
    pass

@dataclasses.dataclass
class StatInitArguments:
    """
    Stats Arguments.
    """
    log_steps: int = 1
    update_system_stats: bool = False
    log_model_steps: int = 1000
    exclude_list: str = r"bias|LayerNorm|layer\.[3-9]|layer\.1(?!1)|layer\.2(?!3)"

class BasicStats:
    """
    Basis Stats class provides a common place for collects long interval stats and step interval
    stats that can be recorded in the various writers provided at the time of calling rebuild()
    in trainer. This class is used as a Singleton pattern via global_stats provided in the
    __init__.py file.
    """
    def __init__(self, args: StatInitArguments, writers=None):
        self.args = args
        self.logger = getlogger(__name__)
        self.reset()
        self.writers = writers

    def rebuild(self, args: StatInitArguments, writers: Iterable):
        """
        Rebuild Stat Args and Writers.
        """
        self.args = args
        self.writers = writers

    def reset(self):
        """
        Reset all stats.
        """
        self.reset_short()
        self.reset_long()

    def reset_short(self):
        """
        Reset step interval stats.
        """
        self.scalars_short = {}
        self.multi_short = {}

    def reset_long(self):
        """
        Reset long interval stats.
        """
        self.scalars_long = {}
        self.multi_long = {}
        self.images = {}
        self.pr = {} # key->(pred, labels)
        self.histogram = {} # key -> vals
        self.embedding = {}

    def update(self, k, v, frequent=False):
        """
        Update step interval and long interval scalar stats.
        """

        if frequent:
            self.scalars_short[k] = v
        else:
            self.scalars_long[k] = v

    def update_multi(self, k, v: dict, frequent=False):
        """
        Update step interval and long interval multiple scalar stats.
        """
        if frequent:
            self.multi_short[k] = v
        else:
            self.multi_long[k] = v

    def update_matplotlib_figure(self, fig, tag):
        """
        Update matplotlib figure.
        """
        try:
            from PIL import Image
        except ImportError:
            self.logger.info("Can't import PIL, can't update matplotlib figure")
            return
        import io
        img = None
        with io.BytesIO() as output:
            fig.savefig(output, format="PNG")
            output.seek(0)
            contents = output.getvalue()
            img = Image.open(io.BytesIO(contents))

        image_arr = np.array(img)
        self.update_image(tag, image_arr, 'HWC')

    def update_image(self, k, v, dataformats='HW'):
        """
        Update image.
        Will be logged with infrequent metric.
        """
        self.images[k] = (v, dataformats)

    def update_pr(self, k, preds, labels):
        """
        Update pr curve stats.
        Only binary classification
        preds = probabilities
        """
        self.pr[k] = (preds, labels)

    def update_histogram(self, k, vals, extend=False):
        """
        Update histogram stats.
        """
        if not extend or k not in self.histogram:
            self.histogram[k] = vals
        else:
            self.histogram[k] = torch.cat((self.histogram[k], vals))

    def update_embedding(self, k, embs, labels):
        """
        Update embeddings.
         Used to project embeddings with corresponding labels (numerical).
        """
        self.embedding[k] = (embs, labels)

    def log_stats(self, step, force=False):

        if (step % self.args.log_steps == 0) or force:
            self.logger.debug(f'logging short stats for step {step}')
            if self.args.update_system_stats:
                self.update_system_stats()
            for writer in self.writers:
                for k, v in self.scalars_short.items():
                    writer.log_scalar(k, v, step)
                for k, v in self.multi_short.items():
                    writer.log_multi(k, v, step)

                #writer.flush()
            #print(self.scalars_short)
            self.reset_short()

    def update_system_stats(self):
        """
         Update system stats related to Memory and Compute (CPU and GPUs) usage.
        """
        try:
            process = psutil.Process(os.getpid())
            #RAM
            self.update('system/RAM/memory_used_pct', psutil.virtual_memory().percent, frequent=True)
            self.update('system/RAM/memory_elr_pct', process.memory_info().rss/psutil.virtual_memory().total *100, \
                        frequent=True)
            #CPU
            self.update('system/CPU/pct', psutil.cpu_percent(interval=1), frequent=True)
            #GPU
            if self.args.device.type != 'cpu':
                self.update('system/GPU0/memory_used_pct', \
                            torch.cuda.memory_allocated(device=self.args.device) / \
                            torch.cuda.get_device_properties(self.args.device).total_memory *100, frequent=True)
        except Exception as e: # pylint: disable=broad-except
            self.logger.warning(f'error in update_system_stats : {e}')

    def log_long_stats(self, step):
        """
        Log long interval stats to correponding writers.
        """
        self.logger.debug(f'logging long stats for step {step}')
        for writer in self.writers:
            for k, v in self.scalars_long.items():
                writer.log_scalar(k, v, step)
            for k, v in self.multi_long.items():
                writer.log_multi(k, v, step)
            for k, v in self.images.items():
                writer.log_image(k, v[0], step, dataformats=v[1])
            for k, v in self.pr.items():
                writer.log_pr_curve(k, preds=v[0], labels=v[1], step=step)
            for k, v in self.histogram.items():
                writer.log_histogram(k, v, step)
            for k, v in self.embedding.items():
                writer.log_embedding(tag=k, mat=v[0], labels=v[1], step=step)
        self.reset_long()

    def log_args(self, args):
        """
        Log Arguments to correponding writers.
        """
        self.logger.debug('Logging args to file.')
        for writer in self.writers:
            writer.log_args(args)

    def log_model(self, step, model, force=False, grad_scale=1):
        """
        Log model to correponding writers.
        """
        self.logger.debug('basic - beginning log_model')
        if (step % self.args.log_model_steps == 0) or force:
            self.logger.info(f'force {force}, logging model stats for step {step}')
            flat_weights, flat_grads = self._get_flat_param_vals(model, grad_scale)

            for writer in self.writers:
                self.logger.debug(f'basic - log_model - beginning writer.log_model {type(writer)}')
                writer.log_model(flat_weights, flat_grads, step)
                self.logger.debug(f'basic - log_model - finishing writer.log_model {type(writer)}')

        self.logger.debug('basic - finishing log_model')

    def log_graph(self, model, device):
        """
        Log graph to correponding writers.
        """
        self.logger.debug('logging graph')
        for writer in self.writers:
            writer.log_graph(model, device=device)

    def finish(self):
        """
        Call finish() on all writers.
        """
        for writer in self.writers:
            writer.finish()

    def _get_flat_param_vals(self, model, grad_scale):
        self.logger.debug('basic - beginning _get_flat_param_vals')
        flat_weights = {}
        flat_grads = {}
        for name, param in model.named_parameters():
            if self._exclude_param(name):
                continue
            flat_weights[name] = param.data.view(-1)
            if param.grad is not None:
                flat_grads[name] = param.grad.view(-1)/grad_scale
        self.logger.debug('basic - finishing _get_flat_param_vals')
        return flat_weights, flat_grads

    def _exclude_param(self, param_name):
        if re.search(self.args.exclude_list, param_name):
            return True
        return False
