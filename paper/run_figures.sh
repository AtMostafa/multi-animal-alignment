# !/bin/bash
# Script to run figures
screen -dmS fig2  -c \
'jupyter nbconvert --execute --to fig2.ipynb --inplace fig2.ipynb'
screen -dmS fig3 bash -c \
'jupyter nbconvert --execute --to fig3.ipynb --inplace fig3.ipynb'
screen -dmS S1 bash -c \
'jupyter nbconvert --execute --to S1-mouse-M1-decoding.ipynb --inplace S1-mouse-M1-decoding.ipynb'
screen -dmS S2 bash -c \
'jupyter nbconvert --execute --to S2-decode-behav-corr.ipynb --inplace S2-decode-behav-corr.ipynb'
screen -dmS S3-monkey bash -c \
'jupyter nbconvert --execute --to S3-monkey-behaviour.ipynb --inplace S3-monkey-behaviour.ipynb'
screen -dmS S3-mouse bash -c \
'jupyter nbconvert --execute --to S3-mouse-behaviour.ipynb --inplace S3-mouse-behaviour.ipynb'
screen -dmS S4 bash -c \
'jupyter nbconvert --execute --to S4-decode-example.ipynb --inplace S4-decode-example.ipynb'
screen -dmS S5 bash -c \
'jupyter nbconvert --execute --to S5-dynamics-example.ipynb --inplace S5-dynamics-example.ipynb'
screen -dmS S6 bash -c \
'jupyter nbconvert --execute --to S6-example-cca.ipynb --inplace S6-example-cca.ipynb'
screen -dmS S7-cca bash -c \
'jupyter nbconvert --execute --to S7-cca-bounds.ipynb --inplace S7-cca-bounds.ipynb'
screen -dmS S7-dim bash -c \
'jupyter nbconvert --execute --to S7-dim.ipynb --inplace S7-dim.ipynb'
screen -dmS S7-VAF bash -c \
'jupyter nbconvert --execute --to S7-VAF.ipynb --inplace S7-VAF.ipynb'
screen -dmS S8 bash -c \
'jupyter nbconvert --execute --to S8-Natcomm-paper-comparison.ipynb --inplace S8-Natcomm-paper-comparison.ipynb'
screen -dmS S9 bash -c \
'jupyter nbconvert --execute --to S9-rnn.ipynb --inplace S9-rnn.ipynb'
