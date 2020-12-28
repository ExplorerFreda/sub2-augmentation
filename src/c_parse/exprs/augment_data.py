import dotdict
import os
import random
import sys
from pathlib import Path
from augmenters import CParseLengthFreeAugmenter, \
    CParseRandomAugmenter, CParseSynonymAugmenter
from data import PTBDataset
from global_utils import save_result, search_hyperparams


def augment(configs):
    configs = dotdict.DotDict(configs)
    os.system(f'mkdir -p /share/data/lang/users/freda/codebase/DomAdapt/data/augment/{configs.data_name}')
    orig_train_path = f'/share/data/lang/users/freda/codebase/DomAdapt/data/{configs.data_name}/train.trees'
    aug_train_path = f'/share/data/lang/users/freda/codebase/DomAdapt/data/augment/' \
        f'{configs.data_name}/train.trees.{configs.augmenter_type}.{configs.augment_idx}'
    orig_dataset = PTBDataset(orig_train_path, use_spans=False)
    if configs.augmenter_type == 'sub':
        augmenter = CParseLengthFreeAugmenter(dataset=orig_dataset)
    elif configs.augmenter_type == 'synonym':
        augmenter = CParseSynonymAugmenter(dataset=orig_dataset)
    elif configs.augmenter_type == 'random':
        augmenter = CParseRandomAugmenter(dataset=orig_dataset)
    else:
        raise Exception(f'Augmenter {configs.augmenter_type} not supported.')
    random.seed(configs.augment_idx + 115)
    aug_dataset = augmenter.augment()
    with open(aug_train_path, 'w') as fout:
        for tree in aug_dataset.trees:
            tree_str = ' '.join(tree.pformat().replace('\n', ' ').split())
            print(tree_str, file=fout)
        fout.close()


def run_trainer(configs):
    configs = dotdict.DotDict(configs)
    os.chdir('../3rdparty/c-parser')
    command = f'''python src/main.py train \
    --train-path-form-vocab /share/data/lang/users/freda/codebase/DomAdapt/data/all/train.trees \
    --train-path /share/data/lang/users/freda/codebase/DomAdapt/data/{configs.data_name}/train.trees \
    --dev-path /share/data/lang/users/freda/codebase/DomAdapt/data/{configs.data_name}/dev.trees \
    --model-path-base {configs.prefix}-{configs.augmenter_type} \
    --bert-model {configs.bert_model} --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 \
    --use-bert --evalb-dir EVALB_SPMRL/ --no-bert-do-lower-case
    '''
    os.system(command)


meta_configs = dotdict.DotDict(
    {
        'data_name': {
            'values': [
                'foreebank/eng', 
                'foreebank/fre', 
                'genia-dist/division', 
                'nxt-switchboard/ptb-style', 
                'qbank'
            ], 
            'flag': 'data'
        },
        'prefix': {
            'values': [
                'models/benepar/foreebank_e',
                'models/benepar/foreebank_f',
                'models/benepar/genia',
                'models/benepar/swbd',
                'models/benepar/qbank'
            ], 
            'flag': 'data'
        },
        'bert_model': {
            'values': ['xlm-roberta-base'],
            'flag': None
        },
        'augment_idx': {
            'values': list(range(20)),
            'flag': None
        },
        'augmenter_type': {
            'values': ['sub', 'random', 'synonym'],
            'flag': None
        },
    }
)
all_configs = list(search_hyperparams(dict(), meta_configs))
print(len(all_configs), 'configs generated.')
idx = int(sys.argv[1])
augment(all_configs[idx])

# for lauching exprs
running_command = '''
    for cnt in {0..0} ; do for idx in {0..299} ; do
    sbatch -p speech-cpu-long \
        -x cpu5,cpu13 \
        --output ../logs/c-parse-augment/$idx.out \
        --error ../logs/c-parse-augment/$idx.err \
        --open-mode append \
        -J cparse_a_$idx \
        -d singleton -c2 \
        launch.sh c_parse.exprs.augment_data $idx
    done 
    done 
'''