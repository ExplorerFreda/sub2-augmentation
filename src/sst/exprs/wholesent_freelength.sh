#!/bin/bash 
source ~/.bashrc
conda activate sub-env
python -m sst.exprs.wholesent_freelength $1
