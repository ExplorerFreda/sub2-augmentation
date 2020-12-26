import dotdict
import os
import submitit
import sys
from pathlib import Path
from agnews.train import train
from global_utils import save_result, search_hyperparams, slurm_job_babysit


meta_configs = dotdict.DotDict(
    {
        'tagset_size': {
            'values': [4],
            'flag': None
        },
        'data_path': {
            'values': [
                '../data/ag_news/{split}_c.txt', 
                '../data/ag_news/{split}_c.txt', 
                '../data/ag_news/{split}_cl.txt', 
                '../data/ag_news/{split}_c.txt', 
                '../data/ag_news/{split}_cl.txt', 
                '../data/ag_news/{split}_pc.txt', 
                '../data/ag_news/{split}_pcl.txt', 
                '../data/ag_news/{split}_pc.txt', 
                '../data/ag_news/{split}_pcl.txt', 
                '../data/ag_news/{split}_c.txt', 
                '../data/ag_news/{split}_c.txt', 
            ],
            'flag': 'data'
        },
        'learning_rate': {
            'values': [0.0005],
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
            'values': ['xlm-roberta-large'],
            'flag': 'pretrained-model'
        },
        'device': {
            'values': ['cuda'],
            'flag': None
        },
        'hidden_dim': {
            'values': [512],
            'flag': None
        },
        'dropout_p': {
            'values': [0.2],
            'flag': None
        },
        'fine_tune': {
            'values': [False],
            'flag': 'fine-tune'
        },
        'batch_size': {
            'values': [64],
            'flag': 'fine-tune'
        },
        'epochs': {
            'values': [20],
            'flag': None
        },
        'validation_per_epoch': {
            'values': [4],
            'flag': 'pretrained-model'
        },
        'seed':{
            'values': [116],
            'flag': 'global-seed'
        },
        'tmp_path':{
            'values': [f'../tmp'],
            'flag': None
        },
        'augment': {
            'values': [
                False, True, True, 'free-length', 'free-length',
                True, True, 'free-length', 'free-length', 'synonym',
                'random'
            ],
            'flag': 'data'
        },
        'use_spans': {
            'values': [False],
            'flag': None
        },
        'use_attn': {
            'values': [True],
            'flag': None
        }
    }
)
all_configs = list(search_hyperparams(dict(), meta_configs))
print(len(all_configs), 'configs generated.')
idx = int(sys.argv[1])
results = [train(x) for x in all_configs[idx:idx+1]]
print(results[0])
os.system(f'mkdir -p ../results/agnews/')
save_result(results, '../results/agnews/attn_frozen.json')