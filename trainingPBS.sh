#!/bin/sh

#PBS -S /bin/bash
#PBS -q cuda
#PBS -N ifml-pytorch
#PBS -l nodes=denbi1-int
#PBS -l walltime=4:00:00
#PBS -e ~/ifml/qsub/stderr.err
#PBS -o ~/ifml/qsub/stdout.log

source ~/.bashrc
conda activate pytorch
cd ~/ifml/ifml_project
CUDA_VISIBLE_DEVICES=0 python3 ~/ifml/ifml_project/jay.py
