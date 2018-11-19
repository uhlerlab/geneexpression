#!/bin/bash

savedir="./results/exp4/"
mkdir $savedir

CUDA_VISIBLE_DEVICES=0 python main.py -bs 128 -sd $savedir -iter 10000 -lamb0 0.01 -lamb1 0.1 -lamb2 1 -lambG2 0.1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "ubOT_ZF_exp4" > "${savedir}log.txt"
