#!/bin/sh

#PBS -N ifml-pytorch
#PBS -l ncpus=4
#PBS -V
#PBS -M jakob@kreft-mail.de
#PBS -m bea
#PBS -e ~/ifml/qsub/stderr.err
#PBS -o ~/ifml/qsub/stdout.log

source ~/.bashrc
conda activate pytorch
cd ~/ifml/ifml_project
python3 ~/ifml/ifml_project/jay.py