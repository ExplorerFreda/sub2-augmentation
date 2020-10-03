import dotdict
import os
import submitit
import sys
from pathlib import Path
from pos.train import train
from pos.utils import extra_low_resource_language_list_ud26
from global_utils import save_result, search_hyperparams, slurm_job_babysit


meta_configs = dotdict.DotDict(
    {
        'lang': {
            'values': extra_low_resource_language_list_ud26,
            'flag': None
        },
        'tagset_path': {
            'values': ['../data/ud-treebanks-v2.6/tags.txt'],
            'flag': None
        },
        'data_path': {
            'values': [
                '../data/ud-treebanks-v2.6/*/{lang}-ud-{split}*.conllu'
            ],
            'flag': None
        },
        'learning_rate': {
            'values': [0.001],
            'flag': 'optimizer'
        },
        'min_lr': {
            'values': [5e-6],
            'flag': None
        },
        'optimizer': {
            'values': ['Adam'],
            'flag': 'optimizer'
        },
        'model_name': {
            'values': ['xlm-roberta-large', 'bert-base-multilingual-cased'],
            'flag': None
        },
        'device': {
            'values': ['cuda'],
            'flag': None
        },
        'hidden_dim': {
            'values': [32, 64, 128, 256, 512],
            'flag': None
        },
        'dropout_p': {
            'values': [0.0, 0.2],
            'flag': None
        },
        'fine_tune': {
            'values': [False],
            'flag': 'fine-tune'
        },
        'batch_size': {
            'values': [32],
            'flag': 'fine-tune'
        },
        'epochs': {
            'values': [100],
            'flag': None
        },
        'validation_per_epoch': {
            'values': [4],
            'flag': 'pretrained-model'
        },
        'seed':{
            'values': [115],
            'flag': 'global-seed'
        },
        'tmp_path':{
            'values': [f'{str(Path.home())}/tmp'],
            'flag': None
        },
        'augment': {
            'values': [True, False],
            'flag': None
        }
    }
)
all_configs = list(search_hyperparams(dict(), meta_configs))

log_folder = '~/logs/postag_logs/'
os.system(f'mkdir -p {log_folder}')
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=4320, slurm_partition=sys.argv[1],
    cpus_per_task=8, gpus_per_node=2, nodes=1, slurm_mem='128G',
    slurm_array_parallelism=2048
)
jobs = executor.map_array(train, all_configs)
from IPython import embed; embed(using=False)  # for babysit jobs
result = [job.result() for job in jobs]
os.system(f'mkdir -p ../result/pos_tagging/extra-low_resource')
for model_name in meta_configs.model_name:
    for augment in meta_configs.augment:
        sub_result = filter(
            lambda x: x[0].model_name == model_name and x[0].augment \
                == augment,
            result
        )
        save_result(
            sub_result, 
            f'../result/pos_tagging/extra-low_resource/' \
            f'{model_name}-{augment}.json'
        )
