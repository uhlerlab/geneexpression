#!/bin/bash

savedir="./results/exp3/"
mkdir $savedir

python get_rho.py -bs 128 -sd $savedir -env "ubOT_NUM_perm_exp3"
