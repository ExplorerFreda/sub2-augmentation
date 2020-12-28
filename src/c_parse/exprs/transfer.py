import dotdict
import os
import random
import sys
from glob import glob
from pathlib import Path

from augmenters import CParseLengthFreeAugmenter, \
    CParseRandomAugmenter, CParseSynonymAugmenter
from data import PTBDataset
from global_utils import save_result, search_hyperparams


def run_trainer(configs):
    configs = dotdict.DotDict(configs)
    os.chdir('../3rdparty/c-parser')
    os.system(f'mkdir -p {os.path.dirname(configs.prefix)}')
    num_existing_model = len(glob(configs.prefix + '_dev=*.pt'))
    if num_existing_model == 0:
        os.system(f'cp {configs.base_model} {configs.prefix}_dev=0.00.pt')
    elif num_existing_model > 1:
        raise Exception('Error: multiple model exists')
    command = f'''MKL_THREADING_LAYER=GNU python src/main.py train \
    --train-path-form-vocab /share/data/lang/users/freda/codebase/DomAdapt/data/all/train.trees \
    --train-path /share/data/lang/users/freda/codebase/DomAdapt/data/{configs.data_name}/train.trees \
    --dev-path /share/data/lang/users/freda/codebase/DomAdapt/data/{configs.data_name}/dev.trees \
    --model-path-base {configs.prefix} \
    --bert-model {configs.bert_model} --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 \
    --use-bert --evalb-dir EVALB_SPMRL/ --no-bert-do-lower-case
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
                'models/transfer/foreebank_e',
                'models/transfer/foreebank_f',
                'models/transfer/genia',
                'models/transfer/swbd',
                'models/transfer/qbank'
            ], 
            'flag': 'data'
        },
        'bert_model': {
            'values': ['xlm-roberta-base'],
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
    for cnt in {0..3} ; do for idx in {0..4} ; do
    sbatch -p speech-gpu \
        -x gpu-g1,gpu-g36 \
        --output ../logs/c-parse/transfer_$idx.out \
        --error ../logs/c-parse/transfer_$idx.err \
        --open-mode append \
        -J cparse_tr_$idx \
        -d singleton -c1 \
        launch.sh c_parse.exprs.transfer $idx
    done 
    done 
'''