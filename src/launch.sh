#!/bin/bash 
echo $SLURMD_NODENAME
echo $(hostname)
echo $CUDA_VISIBLE_DEVICES
source ~/.bashrc
conda activate sub-env
python -m $1 $2
