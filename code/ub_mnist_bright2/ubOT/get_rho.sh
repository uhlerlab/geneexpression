#!/bin/bash

savedir="./results/exp4/"
mkdir $savedir

python get_rho.py -bs 128 -sd $savedir -env "ubOT_mnist_bri2_exp4"
