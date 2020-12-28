import dotdict
import os
import sys
from pathlib import Path
from global_utils import save_result, search_hyperparams


def run_trainer(configs):
    configs = dotdict.DotDict(configs)
    os.chdir('../3rdparty/c-parser')
    command = f'''python src/main.py train \
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
    }
)
all_configs = list(search_hyperparams(dict(), meta_configs))
print(len(all_configs), 'configs generated.')
idx = int(sys.argv[1])
run_trainer(all_configs[idx])

# for lauching exprs
running_command = '''
    for cnt in {0..2} ; do for idx in {0..4} ; do
    sbatch -p speech-gpu -x gpu-g1,gpu-g36 \
        --output ../logs/c-parse/orig_$idx.out \
        --error ../logs/c-parse/orig_$idx.err \
        --open-mode append \
        -J cparse_o_$idx \
        -d singleton -c1 \
        launch.sh c_parse.exprs.orig $idx
    done 
    done 
'''