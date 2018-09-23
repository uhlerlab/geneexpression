#!/bin/bash

savedir="./results/exp1/"
mkdir $savedir

python main.py -bs 128 -sd $savedir -iter 2000 -lamb0 0.03 -lamb1 0.01 -lamb2 1 -lambG 1 -lambG2 1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "ubOT_NUM_smooth_3_exp1" > "${savedir}log.txt"
