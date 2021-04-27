# 0.3.4
* Dependencies split for installation (i.e. pip install marlin[models,plugins])

# 0.3.3
* MarlinAutoModel fix 2 & ddp-amp-apex trainer backend

# 0.3.2
* MarlinAutoModel bug fix when using from_pretrained()

# 0.3.1
* rouge-score added as a required dependency to run any plugins currently

# 0.3.0
* Plugins
* Native Apex
* Turing Automodel

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