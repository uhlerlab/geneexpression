#!/bin/bash

savedir="./results/exp3/"
mkdir $savedir

python main.py -bs 128 -sd $savedir -iter 5000 -lamb0 0.01 -lamb1 .01 -lamb2 1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "ubOT_mnist_perm_adj70_exp3" > "${savedir}log.txt"
