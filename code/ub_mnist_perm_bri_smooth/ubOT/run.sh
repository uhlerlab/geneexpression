#!/bin/bash

savedir="./results/exp12/"
mkdir $savedir

python main.py -bs 128 -sd $savedir -bri 0.7 -iter 1500 -lamb0 0.03 -lamb1 0.01 -lamb2 1 -lambG2 0.01 -psi "JS" -citer 2 -lrG 0.0001 -lrD 0.0001 -env "ubOT_mnist_bri_smooth_exp12" > "${savedir}log.txt"
