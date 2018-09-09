#!/bin/bash

savedir="./results/exp4/"
mkdir $savedir

python main.py -bs 128 -sd $savedir -iter 5000 -lamb0 0.1 -lamb1 0.02 -lamb2 1 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "ubOT_mnist_bri2_exp4" > "${savedir}log.txt"
