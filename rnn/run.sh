# !/bin/bash

export seed_str='1000010 1000011 1000012 1000013 1000014 1000015 1000016 1000017 1000018 1000019'
export seed_str='0 1 2 3 4 5 6 7 8 9'
export sds=($seed_str)
export nseeds=${#sds[@]}
export threads=4
export incr=$((nseeds / threads))

for ((i=0;i<$threads;i++)); 
do
    export i=$i
    screen -dmS run$i bash -c \
    'seeds=($seed_str)
    for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
    do
        for sim_number in $(seq 19 34)
        do
            python3 simulation/run_pipeline.py 100001$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos
            python3 save_pca.py $sim_number
            python3 simulation/run_pipeline.py 100002$seed $sim_number -c config_$sim_number.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_exec_1000010_$sim_number
        done
    done'
done

# for ((i=0;i<$threads;i++)); 
# do
#     export i=$i
#     screen -dmS run$i bash -c \
#     'seeds=($seed_str)
#     for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
#     do
#         # python3 simulation/run_pipeline.py $seed 1 -c config_1.yaml -file dataset_chewie_bl0pos
#         # python3 simulation/run_pipeline.py $seed 4 -c config_2.yaml -file dataset_chewie_bl0pos #gin=0.01
#         # python3 simulation/run_pipeline.py $seed 9 -c config_3.yaml -file dataset_chewie_bl0pos 
#         # python3 simulation/run_pipeline.py $seed 10 -c config_3.yaml -file dataset_chewie_bl0vel 
#         # python3 simulation/run_pipeline.py $seed 12 -c config_1.yaml -file dataset_chewie_bl0pos
#         python3 simulation/run_pipeline.py $seed 16 -c config_1.yaml -file dataset_chewie_bl0pos
#     done'
# done

# python3 simulation/save_pca.py

# export seed_str='1000020 1000021 1000022 1000023 1000024 1000025 1000026 1000027 1000028 1000029'
# export sds=($seed_str)
# export nseeds=${#sds[@]}
# export threads=3
# export incr=$((nseeds / threads))

# for ((i=0;i<$threads;i++)); 
# do
#     export i=$i
#     screen -dmS run$i bash -c \
#     'seeds=($seed_str)
#     for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
#     do
#         # python3 simulation/run_pipeline.py $seed 2 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_1000010_1
#         # python3 simulation/run_pipeline.py $seed 3 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_exec_1000010_1
#         # python3 simulation/run_pipeline.py $seed 5 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_exec_1000010_4
#         # python3 simulation/run_pipeline.py $seed 6 -c config_1.yaml -file dataset_chewie_bl0pos_onehot -cca pca_exec_1000010_1
#         # python3 simulation/run_pipeline.py $seed 7 -c config_1.yaml -file dataset_chewie_bl0pos_onehot -cca pca_1000010_1
#         # python3 simulation/run_pipeline.py $seed 8 -c config_2.yaml -file dataset_chewie_bl0pos_onehot -cca pca_exec_1000010_1
#         # python3 simulation/run_pipeline.py $seed 11 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_1000010_1
#         # python3 simulation/run_pipeline.py $seed 13 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_1000010_12
#         # python3 simulation/run_pipeline.py $seed 14 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_thresh_1000010_12
#         # python3 simulation/run_pipeline.py $seed 15 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_thresh_exec_1000010_12
#         python3 simulation/run_pipeline.py $seed 17 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_thresh_1000010_16
#         # python3 simulation/run_pipeline.py $seed 18 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_movement_on_thresh_exec_1000010_16
#     done'
# done

# python3 simulation/run_pipeline.py 20 2 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_1000010_1
#should work: gin_0.01 vs onehot, 
#hard if you look at CCs 1-4
# 1: base  
# 4: gin=0.01  
# 2: base             cca prep_exec     works
# 3: base             cca exec          okay
# 5: base             cca exec gin=0.01 okay
# 6: onehot           cca exec          bad
# 7: onehot           cca prep_exec     
# 8: onehot,gin=0.01  cca exec          very bad
# 9: gin=0.01, noise
