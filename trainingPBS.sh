#!/bin/sh

#PBS -S /bin/bash
#PBS -q cuda
#PBS -N ifml-pytorch
#PBS -l nodes=denbi2-int
#PBS -l ncpus=4
#PBS -l walltime=8:00:00:00
#PBS -M jakob@kreft-mail.de
#PBS -m bea
#PBS -e ~/ifml/qsub/stderr.err
#PBS -o ~/ifml/qsub/stdout.log

source ~/.bashrc
conda activate pytorch
cd ~/ifml/ifml_project
CUDA_VISIBLE_DEVICES=0 python3 ~/ifml/ifml_project/jay.py