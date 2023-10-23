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
        # simulate standard networkss
        python3 simulation/run_pipeline.py 100$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos
        python3 save_pca.py 1000 1

        for sim in 1 2 3 4
        do
            #simulate constrained networks
            python3 simulation/run_pipeline.py 1000$seed $sim -c config_${sim}.yaml -file dataset_chewie_bl0pos -cca pca_1000_1
        done

    done'
done
