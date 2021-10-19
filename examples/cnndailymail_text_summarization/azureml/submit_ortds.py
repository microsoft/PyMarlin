from azureml.core import Experiment, Workspace, ScriptRunConfig, Datastore
from azureml.core.compute import  AmlCompute
from azureml.core.runconfig import MpiConfiguration

# put your AML workspace config.json in this directory!
ws = Workspace.from_config()
ws_details = ws.get_details()
ds = Datastore(ws, 'ws2_ds')

gpu_compute_target = AmlCompute(workspace=ws, name='LoRA-ND')
print(gpu_compute_target.status.serialize())

from azureml.core import Dataset
from azureml.data import OutputFileDatasetConfig

# create input/output datasets
def get_input_dataset(datastore, path_on_datastore, dataset_name):
    dataset = Dataset.File.from_files(path=[(datastore, path_on_datastore)])
    return dataset.as_named_input(dataset_name).as_download()

def get_output_dataset(datastore, path_on_datastore, dataset_name):
    return OutputFileDatasetConfig(destination=(datastore, path_on_datastore), name=dataset_name).as_mount()

def get_args(outputSuffix="deepspeed_ort_amp_nopadding_v100_8"):
    all_params_default = [
        '--data_path', get_input_dataset(ds, f'datasets/cnn_dm/preprocessed/bart/', "data_path"),
        '--config_path', 'config-ort.yaml',
    ]

    return all_params_default

from azureml.core import Environment

# Creates the environment inside a Docker container.
pytorch_env = Environment(name='pymarlin-ort-ds')
pytorch_env.docker.enabled = True
# docker file in this directory built for your convenience

pytorch_env.docker.base_image = "pymarlin/pymarlin.cuda11.1"
pytorch_env.python.user_managed_dependencies = True
pytorch_env.python.interpreter_path = '/opt/miniconda/bin/python'

mpi = MpiConfiguration()
# NDv2, 8 GPU's per node
mpi.process_count_per_node = 8
mpi.node_count = 1

# ds.upload_files(['local path to preprocessed data'], 'datasets/cnn_dm/preprocessed/bart')

script = "train.py"
codepath = '..'

config = ScriptRunConfig(source_directory=codepath,
                         script=script,
                         arguments=get_args(),
                         compute_target=gpu_compute_target,
                         environment=pytorch_env,
                         distributed_job_config=mpi)

experiment_name = 'summarization_bart_ort_backend'
experiment = Experiment(ws, name=experiment_name)

run = experiment.submit(config)

run.tag('nodes', f'{mpi.node_count}')
run.tag('process_count_per_node', f'{mpi.process_count_per_node}')

print("Submitted run")
print(f"\n{run.get_portal_url()}")
