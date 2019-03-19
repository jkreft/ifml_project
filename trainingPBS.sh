#!/bin/sh

#PBS -q cuda
#PBS -N ifml-pytorch
#PBS -l nodes=denbi2-int:gpus=1
#PBS -V
#PBS -M jakob@kreft-mail.de
#PBS -m bea
#PBS -e ~/ifml/qsub/stderr.err
#PBS -o ~/ifml/qsub/stdout.log

sh ~/ifml/ifml_project/train.sh