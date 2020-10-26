import collections
import dotdict
import hashlib
import json
import os
import random
import submitit
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmenters import POSTagAugmenter
from data import UniversalDependenciesDataset
from global_utils import search_hyperparams, save_result, load_checkpoint

from pos.models import POSTagger
from pos.train import evaluate_accuracy


log_descripter = 'Epoch {epoch}, loss={curr_loss:.5f}, ' \
    'accu_loss={accu_loss:.5f}. best_dev_acc={best_dev_acc:.1f}%'


def evaluate(configs):
    configs, eval_configs = configs

    # set up
    configs = dotdict.DotDict(configs)
    eval_configs = dotdict.DotDict(eval_configs)
    torch.manual_seed(configs.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(configs.seed)

    # load dataset 
    dataset = UniversalDependenciesDataset(
        configs.data_path.format(lang=eval_configs.lang, split='test'),
        configs.tagset_path
    )
    dataloader = DataLoader(
        dataset, batch_size=configs.batch_size, shuffle=False, 
        collate_fn=dataset.collate_fn
    )

    # build models
    model = POSTagger(
        configs.model_name, dataset.tag2id, configs.device, 
        configs.hidden_dim, configs.dropout_p, configs.fine_tune
    )
    if configs.device == 'cuda':
        model = nn.DataParallel(model)

    # load checkpoint
    hasher = hashlib.md5()
    hasher.update(json.dumps(configs, sort_keys=True).encode('utf-8'))
    checkpoint_path = f'{configs.tmp_path}/{hasher.hexdigest()}.pt'
    assert os.path.exists(checkpoint_path)
    trainable_params = list(
        filter(lambda p: p.requires_grad, model.parameters())
    )
    optimizer = getattr(torch.optim, configs.optimizer)(
        trainable_params, 
        lr=configs.learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.2, min_lr=configs.min_lr
    )
    _, best_checkpoint, best_dev_acc = load_checkpoint(
        checkpoint_path, model, optimizer, lr_scheduler
    )

    # evaluate
    model.load_state_dict(best_checkpoint)
    test_acc = evaluate_accuracy(model, dataloader)

    configs.eval_lang = eval_configs.lang
    return configs, best_dev_acc, test_acc


if __name__ == '__main__':
    from pathlib import Path
    meta_configs = dotdict.DotDict(
        {
            'lang': {
                'values': ['en'],
                'flag': None
            },
            'tagset_path': {
                'values': ['../data/universal_treebanks_v2.0/std/tags.txt'],
                'flag': None
            },
            'data_path': {
                'values': [
                    '../data/universal_treebanks_v2.0/std/'
                    '{lang}/*{split}.conll'
                ],
                'flag': None
            },
            'learning_rate': {
                'values': [0.001, 0.0005],
                'flag': 'optimizer'
            },
            'min_lr': {
                'values': [5e-6],
                'flag': None
            },
            'optimizer': {
                'values': ['Adam', 'Adam'],
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
                'values': [128, 256, 512],
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
                'flag': 'augment'
            }
        }
    )
    meta_eval_configs = dotdict.DotDict(
        {
            'lang': {
                'values': [
                    'de', 'es', 'fr', 'it', 'pt-br', 'sv', 'ja', 'id', 'ko'
                ],
                'flag': None
            }
        }
    )
    all_train_configs = list(search_hyperparams(dict(), meta_configs))
    all_eval_configs = list(search_hyperparams(dict(), meta_eval_configs))
    all_configs = list()
    for x in all_train_configs:
        for y in all_eval_configs:
            all_configs.append((x, y))
    log_folder = '~/logs/postag_logs/'
    os.system(f'mkdir -p {log_folder}')
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=4320, slurm_partition=sys.argv[1],
        cpus_per_task=8, gpus_per_node=2, nodes=1, slurm_mem='128G',
        slurm_array_parallelism=2048
    )
    jobs = executor.map_array(evaluate, all_configs)
    from global_utils import slurm_job_babysit
    from IPython import embed; embed(using=False)
    result = [job.result() for job in jobs]
    os.system(f'mkdir -p ../result/pos_tagging/')
    save_result(result, '../result/pos_tagging/zero-shot-transfer-eval.json')