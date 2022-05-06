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
        # python3 simulation/run_pipeline.py 100003$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100002$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos
        python3 simulation/run_pipeline.py 100002$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos
        python3 save_pca.py 1
        python3 simulation/run_pipeline.py 100003$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_1000020_1
        # python3 save_pca.py 20
        # python3 save_pca.py 24
        # #without centering in canoncorr_torch
        # python3 simulation/run_pipeline.py 100005$seed 1 -c config_20.yaml -file dataset_chewie_bl0pos -cca pca_1000020_20
        # python3 simulation/run_pipeline.py 100005$seed 2 -c config_24.yaml -file dataset_chewie_bl0pos -cca pca_1000020_24
    done'
done