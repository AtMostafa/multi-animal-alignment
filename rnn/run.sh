# !/bin/bash
# Script to run simulations
export seed_str='0 1 2 3 4 5 6 7 8 9'
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
        #simulate standard networks
        python3 simulation/run_pipeline.py 100002$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos
        python3 save_pca.py 1
        #simulate penalised networks
        python3 simulation/run_pipeline.py 100005$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_1000020_1
    done'
done