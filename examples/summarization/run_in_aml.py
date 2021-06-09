from azureml.core import Experiment, Workspace, Datastore, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import MpiConfiguration, RunConfiguration, DEFAULT_GPU_IMAGE

# subscription_id = '42ae47bd-b19b-42c1-b0b9-19fd5be9d51b'
# resource_group = 'bert-base'
# workspace_name = 'SubstrateIntelligenceNLR-WS2'
# gpu_cluster_name = "sriovdedicated1"


# subscription_id = 'ea482afa-3a32-437c-aa10-7de928a9e793'
# resource_group = 'onnx_training'
# workspace_name = 'ort_training_dev'
# gpu_cluster_name = "onnx-training-ib"

subscription_id = 'ed2cab61-14cc-4fb3-ac23-d72609214cfd'
resource_group = 'AMLDataCache'
workspace_name = 'datacachetest'
gpu_cluster_name = "v100"

ws = Workspace(subscription_id, resource_group, workspace_name)
ws_details = ws.get_details()
print('Name:\t\t{}\nLocation:\t{}'
      .format(ws_details['name'],
              ws_details['location']))

# ds = ws.get_default_datastore()
ds = Datastore.register_azure_blob_container(workspace=ws,
                                             datastore_name='t1',
                                             container_name='azureml-blobstore-d6fc2475-ad02-44a7-90ff-88a2a91e66b1',
                                             account_name='substrateintel3704284680',
                                             account_key='',
                                             create_if_not_exists=True
                                             )

print('Datastore name: ' + ds.name,
      'Container name: ' + ds.container_name,
      'Datastore type: ' + ds.datastore_type,
      'Workspace name: ' + ds.workspace.name, sep='\n')

gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
print(gpu_compute_target.status.serialize())

script_name = 'Marlin_Scenarios/bart_ds_ort/train.py'
codepath = '.'

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
        '--config_path', 'Marlin_Scenarios/bart_ds_ort/config-prod.yaml',
        '--tmgr.train_batch_size', 32,
        '--tmgr.gpu_batch_size_limit', 64,
        '--tmgr.val_batch_size', 64,
        '--tmgr.epochs', 3,
        '--tmgr.backend', "sp-amp",
        '--dist',
        '--chkp.save_dir', get_output_dataset(ds, f'hanyu/bart/cnndm_sum/' + outputSuffix + "/ckpts/save_dir", "chkp_save_dir"),
        '--chkp.model_state_save_dir', get_output_dataset(ds, f'hanyu/bart/cnndm_sum/' + outputSuffix + "/ckpts/model_state_save_dir", "model_state_save_dir"),
        '--wrt.tb_log_dir', get_output_dataset(ds, f'hanyu/bart/cnndm_sum/' + outputSuffix + "/tblogs", "tb_log_dir"),
        #
        # '--chkp.load_dir', get_input_dataset(ds, f'hanyu/bart/ckpts/cnndm_sum/deepspeed_test_0/save_dir', "load_dir"),
        '--tm.ort',
        '--tm.deepspeed',
        '--tm.deepspeed_config', 'Marlin_Scenarios/bart_ds_ort/deepspeed/deepspeedConfig.json',

    ]
    return all_params_default


from azureml.core import Environment

# Creates the environment inside a Docker container.
pytorch_env = Environment(name='myEnv')
pytorch_env.docker.enabled = True
pytorch_env.docker.base_image = "bart:cuda11.1.cudnn8.ds.ort.v0.2.ms.hf"
pytorch_env.python.user_managed_dependencies = True
pytorch_env.docker.base_image_registry.address = 'hanyu.azurecr.io'
pytorch_env.docker.base_image_registry.username = 'hanyu'
pytorch_env.docker.base_image_registry.password = ''
pytorch_env.python.interpreter_path = '/opt/miniconda/bin/python'

mpi = MpiConfiguration()
mpi.process_count_per_node = 8  # NC SKU has 4 GPU's per node
mpi.node_count = 1  # scale to the amount of nodes you'd like

config = ScriptRunConfig(source_directory=codepath,
                         script=script_name,
                         arguments=get_args(),
                         compute_target=gpu_compute_target,
                         environment=pytorch_env,
                         distributed_job_config=mpi)

experiment_name = 'hanyu_marlin_bart_seq2seqft_benchmark_1'
experiment = Experiment(ws, name=experiment_name)

run = experiment.submit(config)

run.tag('nodes', f'{mpi.node_count}')
print("Submitted run")
print(f"\n{run.get_portal_url()}")
