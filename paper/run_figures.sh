# !/bin/bash

# Script to run figures
screen -dmSL fig2 bash -c \
'jupyter nbconvert --execute --to fig2.ipynb --inplace fig2.ipynb'
screen -dmSL fig3 bash -c \
'jupyter nbconvert --execute --to fig3.ipynb --inplace fig3.ipynb'
screen -dmSL fig4 bash -c \
'jupyter nbconvert --execute --to fig4.ipynb --inplace fig4.ipynb'
screen -dmSL fig5 bash -c \
'jupyter nbconvert --execute --to figg5.ipynb --inplace fig5.ipynb'

#to run
screen -dmSL S1 bash -c \
'jupyter nbconvert --execute --to figS1-monkey-behavior.ipynb --inplace figS1-monkey-behavior.ipynb'
screen -dmSL S2 bash -c \
'jupyter nbconvert --execute --to figS2-TME.ipynb --inplace figS2-TME.ipynb'
screen -dmSL S3-monkey bash -c \
'jupyter nbconvert --execute --to figS3-example-cca.ipynb --inplace figS3-example-cca.ipynb'
screen -dmSL S4 bash -c \
'jupyter nbconvert --execute --to figS4-decoding.ipynb --inplace figS4-decoding.ipynb'
screen -dmSL S5 bash -c \
'jupyter nbconvert --execute --to figS5-mouse-behavior.ipynb --inplace figS5-mouse-behavior.ipynb'
screen -dmSL S6 bash -c \
'jupyter nbconvert --execute --to figS6-mouse-additional.ipynb --inplace figS6-mouse-additional.ipynb'
screen -dmSL S7-cca bash -c \
'jupyter nbconvert --execute --to figS7-topology.ipynb --inplace figS7-topology.ipynb'
screen -dmSL S8 bash -c \
'jupyter nbconvert --execute --to figS8-RW.ipynb --inplace figS8-RW.ipynb'
screen -dmSL S10 bash -c \
'jupyter nbconvert --execute --to figS10-rnn.ipynb --inplace figS10-rnn.ipynb'
