#!/bin/bash
export seed_str='1000010 1000011 1000012 1000013 1000014 1000015 1000016 1000017 1000018 1000019'
export sds=($seed_str)
export nseeds=${#sds[@]}
export threads=2
export incr=$((nseeds / threads))

for ((i=0;i<$threads;i++)); 
do
    export i=$i
    screen -dmS run$i bash -c \
    'seeds=($seed_str)
    for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
    do
        python3 simulation/run_pipeline.py $seed 3 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_exec_1000010_1
        # python3 simulation/run_pipeline.py $seed 2 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_1000010_1
    done'
done
