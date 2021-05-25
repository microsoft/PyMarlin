# 0.4.2
* Disabling all reduce during gradient accumulation for DDP trainer. Significant improvement in _backward() speed.
* Added bart summarization example

# 0.3.2
* Dependencies split for installation (i.e. pip install marlin[dev])

# 0.3.1
* rouge-score added as a required dependency to run any plugins currently

# 0.3.0
* Native Apex

# 0.2.0
* DDPTrainer has been fixed and achieved parity on GLUE Benchmark
* elr2 renamed to marlin
* train_module renamed to module_interface
* data_module renamed to data_interface
* train_manager renamed to trainer
* trainer renamed to trainer_backend

# 0.1.1
* Adding TNLRv3 seq2seq scenario (MSIT only)
* Gated build pipeline now shows tests results and code coverage (bug-ish)
* fixing tests to include tests instead of test :P 
* Testing auto-packaging with the version bump

# 0.1.0
* Initial release for internal ELR team.
* Trainer, TrainerManager, TrainModule, Scenarios, DataModule, Checkpoiner, ConfigParser and more (see documentation)