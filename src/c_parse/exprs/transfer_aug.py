import dotdict
import os
import sys
from glob import glob
from pathlib import Path
from global_utils import save_result, search_hyperparams


def run_trainer(configs):
    configs = dotdict.DotDict(configs)
    os.chdir('../3rdparty/c-parser')
    num_existing_model = len(glob(f'{configs.prefix}-{configs.augmenter_type}_dev=*.pt'))
    if num_existing_model == 0:
        os.system(f'cp {configs.base_model} {configs.prefix}-{configs.augmenter_type}_dev=0.00.pt')
    elif num_existing_model > 1:
        raise Exception('Error: multiple model exists')
    command = f'''MKL_THREADING_LAYER=GNU python src/main.py train \
    --train-path-form-vocab /share/data/lang/users/freda/codebase/DomAdapt/data/all/train.trees \
    --train-path /share/data/lang/users/freda/codebase/DomAdapt/data/augment/{configs.data_name}/train.trees.{configs.augmenter_type} \
    --dev-path /share/data/lang/users/freda/codebase/DomAdapt/data/{configs.data_name}/dev.trees \
    --model-path-base {configs.prefix}-{configs.augmenter_type} \
    --bert-model {configs.bert_model} --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 \
    --use-bert --evalb-dir EVALB_SPMRL/ --no-bert-do-lower-case --checks-per-epoch 40
    '''
    os.system(command)


meta_configs = dotdict.DotDict(
    {
        'base_model': {
            'values': ['models/benepar/ptb_xlmr_dev=94.29.pt'],
            'flag': False
        },
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
                'models/transfer_aug/foreebank_e',
                'models/transfer_aug/foreebank_f',
                'models/transfer_aug/genia',
                'models/transfer_aug/swbd',
                'models/transfer_aug/qbank'
            ], 
            'flag': 'data'
        },
        'bert_model': {
            'values': ['xlm-roberta-base'],
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
run_trainer(all_configs[idx])

# for lauching exprs
running_command = '''
    for cnt in {0..3} ; do for idx in {0..14} ; do
    sbatch -p speech-gpu -x gpu-g1,gpu-g36 \
        --output ../logs/c-parse/transfer_aug_$idx.out \
        --error ../logs/c-parse/transfer_aug_$idx.err \
        --open-mode append \
        -J cparse_atr_$idx \
        -d singleton -c1 \
        launch.sh c_parse.exprs.transfer_aug $idx
    done 
    done
'''