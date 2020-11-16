import dotdict
import os
import regex
import submitit
import sys
import tempfile
from glob import glob
from pathlib import Path
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


def setup_data(data_path, lang):
    train_files = list(
        filter(lambda x:len(regex.findall(f'/{lang}-', x)) == 0, 
        glob(f'{data_path}/*/*train.conll'))
    )
    dev_files = list(
        filter(lambda x:len(regex.findall(f'/{lang}-', x)) == 0, 
        glob(f'{data_path}/*/*dev.conll'))
    )
    test_files = list(
        filter(lambda x:len(regex.findall(f'/{lang}-', x)) > 0, 
        glob(f'{data_path}/*/*test.conll'))
    )
    tmpdir = tempfile.TemporaryDirectory(prefix='zxl-transfer')
    os.system(f'cat {" ".join(train_files)} > {tmpdir.name}/train.conll')
    os.system(f'cat {" ".join(dev_files)} > {tmpdir.name}/dev.conll')
    os.system(f'cat {" ".join(test_files)} > {tmpdir.name}/test.conll')
    return f'{tmpdir.name}/train.conll', f'{tmpdir.name}/dev.conll', \
        f'{tmpdir.name}/test.conll', tmpdir


def run_parser(configs):
    configs = dotdict.DotDict(configs)
    config_ini = config_template
    configs.path = '-'.join(
        [f'{configs[key].split("/")[-1]}' for key in sorted(configs.keys())]
    )
    configs.train_data, configs.dev_data, configs.test_data, \
        tmpdir = setup_data(configs.data_path, configs.lang)
    training_dataset = UniversalDependenciesDataset(
        configs.train_data, f'{configs.data_path}/tags.txt'
    )
    training_dataset.filter_length_for_model('xlm-roberta-large')
    training_dataset.print(configs.train_data)
    dev_dataset = UniversalDependenciesDataset(
        configs.dev_data, f'{configs.data_path}/tags.txt'
    )
    dev_dataset.filter_length_for_model('xlm-roberta-large')
    dev_dataset.print(configs.dev_data)
    test_dataset = UniversalDependenciesDataset(
        configs.test_data, f'{configs.data_path}/tags.txt'
    )
    test_dataset.filter_length_for_model('xlm-roberta-large')
    test_dataset.print(configs.test_data)

    configs.path = f'{Path.home()}/tmp/dep-par/supar-ud/{configs.path}'
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
    tmpdir.cleanup()
    return results


curr_dir = os.path.dirname(os.path.realpath(__file__))
meta_configs = dotdict.DotDict(
    {
        'lang': {
            'values': ['de', 'es', 'fr', 'it', 'pt', 'sv'],
            'flag': None
        }, 
        'data_path': {
            'values': [
                f'{curr_dir}/../../data/universal_treebanks_v2.0/schuster'
            ],
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
