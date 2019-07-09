#!/bin/bash

savedir="./results/exp4/"
mkdir $savedir

# CUDA_VISIBLE_DEVICES=0 python main.py -bs 128 -sd $savedir -iter 10000 -lamb0 0.01 -lamb1 0.1 -lamb2 1 -lambG2 0.1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "geneexpression_exp4" > "${savedir}log.txt"
CUDA_VISIBLE_DEVICES=1 python conditional_main.py -bs 509 -sd $savedir -iter 10000 -lamb0 0.01 -lamb1 0.1 -lamb2 0.01 -lambG2 1 -psi "JS" -citer 2 -lrG 0.01 -lrD 0.01 -env "nehaenv" > "${savedir}log.txt"


# CUDA_VISIBLE_DEVICES=0 python main.py -bs 128 -sd $savedir -iter 10000 -lamb0 0.01 -lamb1 0.1 -lamb2 1 -lambG2 0.1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 > "${savedir}log.txt"
