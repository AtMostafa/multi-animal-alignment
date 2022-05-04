# !/bin/bash

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
        for sim_number in 31 20 24
        do
            # python3 simulation/run_pipeline.py 100001$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos
            python3 simulation/run_pipeline.py 100002$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos
            python3 save_pca.py $sim_number
            # python3 simulation/run_pipeline.py 100003$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_1000010_$sim_number
            python3 simulation/run_pipeline.py 100005$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_1000010_$sim_number
        done
    done'
done