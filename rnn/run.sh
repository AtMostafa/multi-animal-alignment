# !/bin/bash
# Script to run simulations
export seed_str='0 1 2 3 4 5 6 7 8 9'
# export seed_str='0 1'
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
        # python3 simulation/run_pipeline.py 100002$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100002$seed 3 -c config_3.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100002$seed 4 -c config_4.yaml -file dataset_chewie_bl0pos
        #simulate with nonlinear readout
        # python3 simulation/run_pipeline.py 100002$seed 8 -c config_8.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100002$seed 9 -c config_9.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100002$seed 12 -c config_9.yaml -file dataset_chewie_bl0pos
        python3 save_pca.py 12
        #standard
        # python3 simulation/run_pipeline.py 100002$seed 11 -c config_11.yaml -file dataset_chewie_bl0pos
        #simulate with FORCE
        # python3 simulation/run_pipeline.py 100005$seed 3 -c config_3.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100005$seed 5 -c config_5.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100005$seed 6 -c config_6.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100005$seed 7 -c config_7.yaml -file dataset_chewie_bl0pos
        # python3 simulation/run_pipeline.py 100005$seed 10 -c config_10.yaml -file dataset_chewie_bl0pos
       #simulate penalised networks
        # python3 simulation/run_pipeline.py 100005$seed 1 -c config_1.yaml -file dataset_chewie_bl0pos -cca pca_1000020_1
        # python3 simulation/run_pipeline.py 100005$seed 2 -c config_2.yaml -file dataset_chewie_bl0pos -cca pca_1000020_2
        # python3 simulation/run_pipeline.py 100005$seed 8 -c config_8.yaml -file dataset_chewie_bl0pos -cca pca_1000020_8
        python3 simulation/run_pipeline.py 100005$seed 12 -c config_9.yaml -file dataset_chewie_bl0pos -cca pca_1000020_12
    done'
done

# python3 simulation/run_pipeline.py 1000020 3 -c config_3.yaml -file dataset_chewie_bl0pos
#31: 0.2 noise, 0.01 ginout
#1: basic without reg
#2: basic without reg, 0.2 noise
#3: basic without reg, force
#4: basic with reg
#5: basic without reg, force, alpha = 
#6: basic without reg, force, alpha = 
#7: basic without reg, force, alpha = 
#8: nonlinear without reg
#9: nonlinear with reg
#10:basic without reg, force, gin = 1.0
#11:basic, gin = 1.0


# python3 simulation/run_pipeline.py 1000050 6 -c config_5.yaml -file dataset_chewie_bl0pos -a lr 1.0
# python3 simulation/run_pipeline.py 100002$seed 12 -c config_9.yaml -file dataset_chewie_bl0pos
