import dotdict
import os
import submitit
import sys
import tempfile
from glob import glob
from pathlib import Path

from augmenters import DependencyParsingAugmenter, \
    DependencyParsingBaselineAugmenter, DependencyParsingJointAugmenter
from data import UniversalDependenciesDataset
from global_utils import save_result, search_hyperparams, slurm_job_babysit


config_template = """[Data]
bert = 'xlm-roberta-large'

[Network]
n_embed = 100
n_char_embed = 50
n_feat_embed = 100
n_bert_layers = 4
mix_dropout = .0
embed_dropout = .33
n_lstm_hidden = 400
n_lstm_layers = 3
lstm_dropout = .33
n_mlp_arc = 500
n_mlp_rel = 100
mlp_dropout = .33

[Optimizer]
lr = 2e-3
mu = .9
nu = .9
epsilon = 1e-12
clip = 5.0
decay = .75
decay_steps = 5000
"""

run_command = """ python -m supar.cmds.biaffine_dependency train -b \\
    -d 0 -c {path}/config.ini  -p {path}/model  -f bert \\
    --bert xlm-roberta-large --train {data_train} \\
    --dev {data_dev} --test {data_test}
"""


def run_parser(configs):
    configs = dotdict.DotDict(configs)
    config_ini = config_template
    configs.path = '-'.join([
        f'{str(configs[key]).split("/")[-1]}' for key in sorted(configs.keys())
    ])
    try:
        configs.train_data = glob(f'{configs.data_path}/*train.conllu')[0]
        configs.dev_data = glob(f'{configs.data_path}/*dev.conllu')[0]
        configs.test_data = glob(f'{configs.data_path}/*test.conllu')[0]
    except:
        return None
    tmp_dir = tempfile.TemporaryDirectory(prefix='ud-en')
    # static augmentation: load and augment the training set
    training_dataset = UniversalDependenciesDataset(
        configs.train_data, f'{os.path.dirname(configs.data_path)}/tags.txt'
    )
    if configs.orig_size is not None:
        training_dataset.cut(configs.orig_size)
    training_dataset.filter_length_for_model('xlm-roberta-large')
    configs.train_data = f'{tmp_dir.name}/train.conll'
    training_dataset.print(configs.train_data)
    # cut the dev data
    dev_dataset = UniversalDependenciesDataset(
        configs.dev_data, 
        f'{os.path.dirname(configs.data_path)}/tags.txt'
    )
    if configs.orig_size is not None:
        dev_dataset.cut(configs.orig_size // 10)
    configs.dev_data = f'{tmp_dir.name}/dev.conll'
    dev_dataset.print(configs.dev_data)
    # config saving path
    configs.path = f'{Path.home()}/tmp/dep-par/supar-ud-sub/size-study-noaug/{configs.path}'
    os.system(f'mkdir -p {configs.path}')
    with open(f'{configs.path}/config.ini', 'w') as fout:
        print(config_ini, file=fout)
        fout.close()
    os.chdir('../3rdparty/parser')
    os.system(run_command.format(
        path=configs.path, data_train=configs.train_data,
        data_dev=configs.dev_data, data_test=configs.test_data
    ))
    results = open(f'{configs.path}/model.train.log').readlines()[-4:]
    tmp_dir.cleanup()
    return results


curr_dir = os.path.dirname(os.path.realpath(__file__))
meta_configs = dotdict.DotDict(
    {
        'seed': {
            'values': [115],
            'flag': None
        }, 
        'data_path': {
            'values': [
                f'{curr_dir}/../../data/universal-dependencies-1.2/UD_English'
            ],
            'flag': None
        },
        'orig_size': {
            'values': [10, 50, 100, 500, 1000],
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
