#!/bin/sh

#PBS -S /bin/bash
#PBS -q cuda
#PBS -N ifml-pytorch-cuda
#PBS -l nodes=denbi1-int
#PBS -l walltime=10:00:00

source ~/.bashrc
conda activate pytorch
cd ~/ifml/ifml_project
CUDA_VISIBLE_DEVICES=1 python3 ~/ifml/ifml_project/jay.py
