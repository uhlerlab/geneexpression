#!/bin/bash

savedir="./results/exp1/"
mkdir $savedir

python get_rho.py -bs 128 -sd $savedir -env "ubOT_mnist_perm_exp1"
