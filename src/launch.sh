#!/bin/bash 
echo $SLURMD_NODENAME
echo $(hostname)
source ~/.bashrc
conda activate sub-env
python -m $1 $2
