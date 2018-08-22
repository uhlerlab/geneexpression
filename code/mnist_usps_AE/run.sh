#!/bin/bash

savedir='./results/'
mkdir ${savedir}

for zdim in 30
do  
    for lamb in 0.0000001 0.000005 0.000003
    do
        name="NUM_${zdim}_${lamb}"
        pthfile="${savedir}${name}"
        logfile="${pthfile}.txt"
        python train.py -iter 250 -sf $pthfile -env $name -nz $zdim -lamb ${lamb} > $logfile
    done
    
done

