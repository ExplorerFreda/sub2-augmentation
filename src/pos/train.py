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
from global_utils import search_hyperparams, AverageMeter, save_result, \
    save_checkpoint, load_checkpoint

from pos.models import POSTagger
from pos.utils import high_resource_language_list, low_resource_language_list


log_descripter = 'Epoch {epoch}, loss={curr_loss:.5f}, ' \
    'accu_loss={accu_loss:.5f}. best_dev_acc={best_dev_acc:.1f}%'


def evaluate_accuracy(model, dataloader):
    model.eval()
    device = model.device if isinstance(
        model, POSTagger) else model.module.device
    num, denom = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            targets = torch.tensor(dataloader.dataset.get_tag_ids(
                batch)).long().to(device).view(-1)
            logits = POSTagger.forward_batch(model, batch).view(
                targets.shape[0], -1)
            preds = logits.max(-1)[1]
            correct_pred = (preds == targets).long().sum().item()
            total_items = (targets != -1).long().sum().item()
            num += correct_pred
            denom += total_items
    model.train()
    return float(num) / denom


def train(configs):
    # set up
    configs = dotdict.DotDict(configs)
    torch.manual_seed(configs.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(configs.seed)

    # load dataset 
    dataset = collections.defaultdict(None)
    dataloader = collections.defaultdict(None)
    for split in ['train', 'dev', 'test']:
        dataset[split] = UniversalDependenciesDataset(
            configs.data_path.format(lang=configs.lang, split=split),
            configs.tagset_path
        )
        dataloader[split] = DataLoader(
            dataset[split], batch_size=configs.batch_size,
            shuffle=(split == 'train'), collate_fn=dataset[split].collate_fn
        )
    if configs.augment:
        augmenter = POSTagAugmenter(dataset['train'])

    # build models
    model = POSTagger(
        configs.model_name, dataset['train'].tag2id, configs.device, 
        configs.hidden_dim, configs.dropout_p, configs.fine_tune
    )
    if configs.device == 'cuda':
        model = nn.DataParallel(model)

    # build optimizer
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
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # load checkpoint (if exists)
    hasher = hashlib.md5()
    hasher.update(json.dumps(configs, sort_keys=True).encode('utf-8'))
    checkpoint_path = f'{configs.tmp_path}/{hasher.hexdigest()}.pt'
    if os.path.exists(checkpoint_path):
        start_epoch_id, checkpoint, best_dev_acc = load_checkpoint(
            checkpoint_path, model, optimizer, lr_scheduler
        )
    else:
        start_epoch_id = 0
        best_dev_acc = 0
        checkpoint = model.state_dict()

    # train
    for epoch_id in range(start_epoch_id+1, configs.epochs):
        # augment training set if applicable 
        random.seed(epoch_id * configs.seed)
        if configs.augment:
            augmenter.reset()
            dataset['train'] = augmenter.augment()
            dataloader['train'] = DataLoader(
                dataset['train'], batch_size=configs.batch_size,
                shuffle=True, collate_fn=dataset['train'].collate_fn
            )
        # train model 
        model.train()
        loss_meter = AverageMeter()
        bar = tqdm(dataloader['train'])
        eval_chunk = len(bar) // configs.validation_per_epoch
        if eval_chunk == 0:
            eval_chunk = 1
        for i, batch in enumerate(bar):
            targets = torch.tensor(dataset['train'].get_tag_ids(
                batch)).long().to(configs.device).view(-1)
            logits = POSTagger.forward_batch(model, batch).view(
                targets.shape[0], -1)
            loss = cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), (targets != -1).sum())
            bar.set_description(
                log_descripter.format(
                    epoch=epoch_id, 
                    curr_loss=loss_meter.val,
                    accu_loss=loss_meter.avg,
                    best_dev_acc=best_dev_acc * 100.0
                )
            )
            if (i + 1) % eval_chunk == 0:
                accuracy = evaluate_accuracy(model, dataloader['dev'])
                lr_scheduler.step(accuracy)
                if accuracy > best_dev_acc:
                    best_dev_acc = accuracy
                    checkpoint = model.state_dict()
        # save checkpoint after each epoch
        save_checkpoint(
            path=checkpoint_path, 
            model=model, 
            best_dev_model=checkpoint, 
            best_dev_performance=best_dev_acc,
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            epoch_id=epoch_id
        )
    
    # test 
    model.load_state_dict(checkpoint)
    test_acc = evaluate_accuracy(model, dataloader['test'])

    return configs, best_dev_acc, test_acc


if __name__ == '__main__':
    from pathlib import Path
    meta_configs = dotdict.DotDict(
        {
            'lang': {
                'values': high_resource_language_list + \
                    low_resource_language_list,
                'flag': None
            },
            'tagset_path': {
                'values': ['../data/universal-dependencies-1.2/tags.txt'],
                'flag': None
            },
            'data_path': {
                'values': ['../data/*/*/{lang}-ud-{split}*.conllu'],
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
                'values': [True],
                'flag': 'augment'
            }
        }
    )
    all_configs = list(search_hyperparams(dict(), meta_configs))
    from IPython import embed; embed(using=False)
