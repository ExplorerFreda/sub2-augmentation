import dotdict
import os
import submitit
import sys
from pathlib import Path
from global_utils import save_result, search_hyperparams, slurm_job_babysit


config_template = """[Data]
bert = 'xlm-roberta-large'

[Network]
n_feat_embed = {}
embed_dropout = {}
n_mlp_arc = {}
n_mlp_rel = {}
mlp_dropout = {}

[Optimizer]
lr = {}
mu = .9
nu = .9
epsilon = 1e-12
clip = 5.0
decay = .75
decay_steps = 5000
"""

run_command = """ python -m supar.cmds.simplest_biaffine_dependency train -b \\
    -d 0 -c {path}/config.ini  -p {path}/model  -f bert
"""


def run_parser(configs):
    configs = dotdict.DotDict(configs)
    config_ini = config_template.format(
        configs.n_feat_embed,
        configs.embed_dropout,
        configs.n_mlp_arc,
        configs.n_mlp_rel,
        configs.mlp_dropout,
        configs.lr
    )
    configs.path = '-'.join(
        [f'{configs[key]}' for key in sorted(configs.keys())]
    )
    configs.path = f'{Path.home()}/tmp/dep-par/{configs.path}'
    os.system(f'mkdir -p {configs.path}')
    with open(f'{configs.path}/config.ini', 'w') as fout:
        print(config_ini, file=fout)
        fout.close()
    os.chdir('../3rdparty/parser')
    os.system(run_command.format(path=configs.path))


meta_configs = dotdict.DotDict(
    {
        'n_feat_embed': {
            'values': [128, 256, 512],
            'flag': None
        },
        'embed_dropout': {
            'values': [0, 0.1, 0.2],
            'flag': None
        },
        'n_mlp_arc': {
            'values': [128, 256, 512],
            'flag': None
        },
        'n_mlp_rel': {
            'values': [64, 128],
            'flag': None
        },
        'mlp_dropout': {
            'values': [0, 0.1, 0.2],
            'flag': None
        },
        'lr': {
            'values': [1e-3, 5e-4, 1e-4],
            'flag': None
        }
    }
)


all_configs = list(search_hyperparams(dict(), meta_configs))

log_folder = '~/logs/dep_par/'
os.system(f'mkdir -p {log_folder}')
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=4320, slurm_partition=sys.argv[1],
    cpus_per_task=8, gpus_per_node=1, nodes=1, slurm_mem='128G',
    slurm_array_parallelism=2048
)
jobs = executor.map_array(run_parser, all_configs)
from IPython import embed; embed(using=False)  # for babysit jobs
result = [job.result() for job in jobs]
