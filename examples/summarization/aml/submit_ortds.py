from azureml.core import Experiment, Workspace, Datastore, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import MpiConfiguration, RunConfiguration, DEFAULT_GPU_IMAGE
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument('--image_exists', action='store_true', help='whether to use an existing image from a private ACR')
args = parser.parse_args()

# put your AML workspace JSON and add tenant id in this directory!
with open('./config.json','r') as f:
    aml_config = json.load(f)
tenant_id = aml_config['tenant_id']
ws = Workspace.from_config('./config.json', auth=InteractiveLoginAuthentication(tenant_id))
ws_details = ws.get_details()
print('Name:\t\t{}\nLocation:\t{}'
      .format(ws_details['name'],
              ws_details['location']))

ds = ws.get_default_datastore()
print('Datastore name: ' + ds.name,
      'Container name: ' + ds.container_name,
      'Datastore type: ' + ds.datastore_type,
      'Workspace name: ' + ds.workspace.name, sep='\n')

kv = ws.get_default_keyvault()

gpu_compute_target = ComputeTarget(workspace=ws, name='sriovdedicated1')
print(gpu_compute_target.status.serialize())

script_name = 'train_ortds.py'
codepath = '..'

from azureml.core import Dataset
from azureml.data import OutputFileDatasetConfig


# create input/output datasets
def get_input_dataset(datastore, path_on_datastore, dataset_name):
    dataset = Dataset.File.from_files(path=[(datastore, path_on_datastore)])
    return dataset.as_named_input(dataset_name).as_mount()


def get_output_dataset(datastore, path_on_datastore, dataset_name):
    return OutputFileDatasetConfig(destination=(datastore, path_on_datastore), name=dataset_name).as_mount()


def get_args(outputSuffix="deepspeed_ort_amp_nopadding_v100_8"):
    all_params_default = [
        '--data_path', get_input_dataset(ds, f'krishan/bart/cnn_dm', "data_path"),
        '--config_path', 'config-prod.yaml',
        '--trainer.train_batch_size', 32,
        '--trainer.gpu_batch_size_limit', 32,
        '--trainer.val_batch_size', 64,
        '--trainer.epochs', 3,
        '--trainer.backend', "ddp-amp-apex",
        '--trainer.disable_tqdm', "true", # ugly logging in AML
        '--chkp.save_dir', get_output_dataset(ds, f'jsleep/bart/cnndm_sum/' + outputSuffix + "/ckpts/save_dir", "chkp_save_dir"),
        '--chkp.model_state_save_dir', get_output_dataset(ds, f'jsleep/bart/cnndm_sum/' + outputSuffix + "/ckpts/model_state_save_dir", "model_state_save_dir"),
        '--wrt.tb_log_dir', get_output_dataset(ds, f'jsleep/bart/cnndm_sum/' + outputSuffix + "/tblogs", "tb_log_dir"),
        # '--chkp.load_dir', get_input_dataset(ds, f'jsleep/bart/ckpts/cnndm_sum/deepspeed_test_0/save_dir', "load_dir"),
        '--module.ort',
        '--module.deepspeed',
        '--module.deepspeed_transformer_kernel',
        '--module.deepspeed_config', 'deepspeed_methods/deepspeedConfig.json',
    ]
    return all_params_default


from azureml.core import Environment

# Creates the environment inside a Docker container.
pytorch_env = Environment(name='myEnv')
pytorch_env.docker.enabled = True

if args.image_exists:
    # this is the image built in the Dockerfile local to the file. re-using because it takes > 1.5hrs to build, greater than aml default timeout
    pytorch_env.docker.base_image = "bart:cuda11.1.cudnn8.ds.ort.pymarlin0.2.3"
    pytorch_env.python.user_managed_dependencies = True
    pytorch_env.docker.base_image_registry.address = 'elrsubstrate.azurecr.io'
    pytorch_env.docker.base_image_registry.username = 'elrsubstrate'
    pytorch_env.docker.base_image_registry.password = kv.get_secret('elrsubstrate-acr-password')
    pytorch_env.python.interpreter_path = '/opt/miniconda/bin/python'
else:
    with open("Dockerfile", "r") as f:
        dockerfile=f.read()
    pytorch_env.docker.base_dockerfile = dockerfile

mpi = MpiConfiguration()
#NCv3_24rs - 4 16GB V100 GPU's per node
mpi.process_count_per_node = 4
mpi.node_count = 2

config = ScriptRunConfig(source_directory=codepath,
                         script=script_name,
                         arguments=get_args(),
                         compute_target=gpu_compute_target,
                         environment=pytorch_env,
                         distributed_job_config=mpi)

experiment_name = 'josleep_pymarlin_summarization_bart_ortds'
experiment = Experiment(ws, name=experiment_name)

run = experiment.submit(config)

run.tag('nodes', f'{mpi.node_count}')
run.tag('process_count_per_node', f'{mpi.process_count_per_node}')
run.tag('notes', '2 node with ort+ds')

print("Submitted run")
print(f"\n{run.get_portal_url()}")
