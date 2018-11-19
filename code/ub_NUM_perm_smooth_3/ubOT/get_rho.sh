#!/bin/bash

savedir="./results/exp1/"
mkdir $savedir

python get_rho.py -bs 128 -sd $savedir -env "ubOT_NUM_smooth_3_exp1"
