from tqdm import tqdm
import torch
from typing import Iterable, List, Optional, Union

from pymarlin import SingleProcess
from pymarlin.core import module_interface
from pymarlin.core.trainer_backend import TrainerBackendArguments, OutputCollector, DDPTrainerBackend


class deepspeed_trainer_backend(SingleProcess):

    def init(self, args: TrainerBackendArguments):
        self.args = args
        self.model = self.args.model
        if not self.distributed:
            assert self.args.distributed_training_args.world_size == 1 \
                , 'World size > 1 . Decorate with DDPTrainerBackend'

        # ensure gradient_accumulation will be equal to the one set in deepspeed config json
        if self.args.gradient_accumulation != self.model.model.gradient_accumulation_steps():
            print(f"Warning, self.args.gradient_accumulation ({self.args.gradient_accumulation}) is not equal to gradient_accumulation_steps inside deepspeedConfig.json, adjusting")
            print(f"Warning, setting self.args.gradient_accumulation to {self.model.model.gradient_accumulation_steps()}")
            self.args.gradient_accumulation = self.model.model.gradient_accumulation_steps()

    def train_dl(self, dataloader, callback: module_interface.CallbackInterface):

        epoch_collector = OutputCollector()
        global_step_collector = OutputCollector()
        self.global_step_this_epoch = 0
        # can pass certain stuff as argument instead of passing the entire train module.
        # But will this hinder inheritence as different trainer_backends will need different stuff from train module
        with tqdm(dataloader, unit="batch", disable=self.args.disable_tqdm) as tbatch:
            for i, batch in enumerate(tbatch):
                if (
                        self.args.max_train_steps_per_epoch
                        and self.global_step_this_epoch
                        >= self.args.max_train_steps_per_epoch
                ):
                    break

                tbatch.set_description(f"Global Batch: {self.global_step_completed + 1} ")
                # forward
                outputs = self.model.forward(
                    stage=module_interface.Stage.TRAIN,
                    batch=batch,
                    device=self.args.device,
                    global_step=self.global_step_completed + 1,
                )
                # assume iterable if first return type is not a list
                outputs = [outputs] if type(outputs) == torch.Tensor else outputs

                loss = outputs[0]

                # backward. This will keep on accumulating gradients
                self.model.model.backward(loss)
                # deepspeed model engine must be called each micro step
                self.model.model.step()
                callback.on_end_backward(self.global_step_completed, loss)

                # collect
                epoch_collector.collect(outputs)
                global_step_collector.collect(outputs)

                unscaled_loss = outputs[0].item()
                tbatch.set_postfix(
                    loss=unscaled_loss
                )  # move progress bar to logger later

                self.batches_completed += 1

                if self.batches_completed % self.args.gradient_accumulation == 0:
                    # write global step mean loss to stats
                    self.process_global_step(global_step_collector, callback)

        return epoch_collector.all_outputs

    def process_global_step(self, global_step_collector, callback):
        """Clip gradients and call optimizer + scheduler
        """
        global_step_outputs = global_step_collector.all_outputs
        global_step_mean_loss = (
            global_step_outputs[0].mean().item()
        )
        global_step_collector.reset()
        self.stats.update("loss", global_step_mean_loss, frequent=True)

        self.global_step_completed += 1
        self.global_step_this_epoch += 1

        callback.on_end_train_step(self.global_step_completed, *global_step_outputs)
        self.stats.log_stats(self.global_step_completed)


class deepspeed_dist_trainer_backend(DDPTrainerBackend):

    def init(self, args: TrainerBackendArguments):
        # unpack trainer_backend arguments
        self.args = args
        self.distributed_training_args = args.distributed_training_args

        self.trainer_backend.init(args)
