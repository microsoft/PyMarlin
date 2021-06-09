import os

from pymarlin import Trainer
from pymarlin.utils.checkpointer.checkpoint_utils import Checkpoint


class deepspeed_Trainer(Trainer):

    def save_checkpoint(self, force=False) -> None:
        # deepspeed will require all processes to call save_checkpoint method
        ckpt_id = str(self.trainer_backend.get_state()["global_step_completed"])
        self.module.model.save_checkpoint(os.path.join(self.args.checkpointer_args.save_dir, self.module.DEEPSPEED_CKPT_PREFIX), ckpt_id)

        if self.is_main_process:  # only main process should checkpoint
            checkpoint_state = Checkpoint(
                module_interface_state=self.module.get_state(),
                trainer_state=self.get_state(),
                trainer_backend_state=self.trainer_backend.get_state()
            )
            self.checkpointer.save(checkpoint_state, self.last_epoch, force)

    def save_model_checkpoint(self) -> None:
        if self.args.checkpointer_args.checkpoint and (self.args.checkpointer_args.model_state_save_dir is not None):
            ckpt_id = str(self.trainer_backend.get_state()["global_step_completed"])
            self.module.model.save_checkpoint(os.path.join(self.args.checkpointer_args.model_state_save_dir, self.module.DEEPSPEED_CKPT_PREFIX), ckpt_id)
