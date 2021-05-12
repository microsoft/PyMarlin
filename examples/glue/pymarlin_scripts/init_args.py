import dataclasses

@dataclasses.dataclass
class DataInterfaceArguments:
    input_dir: ""
    output_dir: ""

@dataclasses.dataclass
class DataProcessorArguments:
    task: None
    max_seq_len: 128
    no_labels: False
    set_type: "train"
    tokenizer: "bert-base-uncased"

@dataclasses.dataclass
class ModelArgs:
    model_name: "bert"
    encoder_key: "bert"
    hf_model: "bert-base-uncased"
    model_file: None
    model_wts_path: None
    model_config_file: "config.json"
    model_config_path: None
    get_latest_ckpt: True
    num_labels: None

@dataclasses.dataclass
class ModuleInterfaceArguments:
    operation: "train"
    trainer_backend: "SingleProcess"
    fp16: False
    trainpath: ""
    valpath: ""
    output_dir: ""
    task: None
    max_lr: 0.00004 # Maximum learning rate.
    warmup_prop: 0.1 # % of steps
    num_files: -1
    no_labels: False
    log_level: "INFO"
    model_args: ModelArgs = ModelArgs

@dataclasses.dataclass
class DistillArgs:
    student_layers: "[0-6-11]"
    loss_types: "[logits-attentions]"
    loss_weights: "[1-1]"
    temperature: 1
    width_shrinkage: 0