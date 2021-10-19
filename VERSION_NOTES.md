# Version Notes:
## 0.3.0
* Added DpSgd backend support

## 0.2.8
* Incremented the dependency to torch<=1.9.1

## 0.2.7
* Adding torch<=1.9 as required dependency

## 0.2.6
* Adding support for parsing multi-level args from commandline and params

## 0.2.5
* Adding support for directories with config path (only one file in directory)

## 0.2.4
* Fixed bug where DDP all-reduce was not working

## 0.2.3
* Unbound azureml-core version

## 0.2.2
* Plugins bug fix

## 0.2.0
* Adding plugins: SeqClassification, NER, Seq2Seq
* --params json input
* DDP allreduce optimization

## 0.1.1
* Tests & Lint Pipeline
* Documentation Pipeline
* PyPi Pipeline

## 0.1.0
* Initial release
* Trainer, TrainerBackend, ModuleInterface, DataProcessor/Interface, ConfigParser and more (see docs)