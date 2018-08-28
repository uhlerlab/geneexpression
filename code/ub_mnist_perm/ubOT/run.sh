#!/bin/bash

savedir="./results/exp1/"
mkdir $savedir

python main.py -bs 128 -sd $savedir -iter 5000 -lamb0 1 -lamb1 .01 -lamb2 1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "ubOT_mnist_perm_exp1" > "${savedir}log.txt"
