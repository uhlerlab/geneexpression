#!/bin/bash

for idx in {9..12}
do
savedir="./results/exp${idx}/"
python get_rho.py -bs 128 -sd $savedir -env "ubOT_mnist_bri_smooth_exp${idx}"
done
