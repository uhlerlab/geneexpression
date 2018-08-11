#!/bin/bash

savedir="./results/"
mkdir $savedir

python main.py -sd $savedir -iter 1000 -psi "JS" -citer 5 -lrG 0.0001 -lrD 0.0001 > "${savedir}log.txt"
