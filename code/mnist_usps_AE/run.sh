#!/bin/bash

savedir='./results/'
mkdir ${savedir}

for zdim in 20
do  
    for lamb in 0.1 0.01 0.001
    do
        name="NUM_${zdim}_${lamb}"
        pthfile="${savedir}${name}"
        logfile="${pthfile}.txt"
        python train.py -iter 100 -sf $pthfile -env $name -nz $zdim -lamb ${lamb} > $logfile
    done
    
done

