# !/bin/bash
# Script to run figures
screen -dmS fig2  -c \
'jupyter nbconvert --execute --to fig2.ipynb --inplace fig2.ipynb'
screen -dmS fig3 bash -c \
'jupyter nbconvert --execute --to fig3.ipynb --inplace fig3.ipynb'
screen -dmS fig4 bash -c \
'jupyter nbconvert --execute --to fig4.ipynb --inplace fig4.ipynb'
screen -dmS fig5 bash -c \
'jupyter nbconvert --execute --to figg5.ipynb --inplace fig5.ipynb'

# screen -dmS S1 bash -c \
# 'jupyter nbconvert --execute --to figS1-monkey-behavior.ipynb --inplace figS1-monkey-behavior.ipynb'
screen -dmS S2 bash -c \
'jupyter nbconvert --execute --to figS2-TME.ipynb --inplace figS2-TME.ipynb'
screen -dmS S3-monkey bash -c \
'jupyter nbconvert --execute --to figS3-example-cca.ipynb --inplace figS3-example-cca.ipynb'
# screen -dmS S4 bash -c \
# 'jupyter nbconvert --execute --to figS4-decoding.ipynb --inplace figS4-decoding.ipynb'
# screen -dmS S5 bash -c \
# 'jupyter nbconvert --execute --to figS5-mouse-behavior.ipynb --inplace figS5-mouse-behavior.ipynb'
# screen -dmS S6 bash -c \
# 'jupyter nbconvert --execute --to figS6-mouse-additional.ipynb --inplace figS6-mouse-additional.ipynb'
screen -dmS S7-cca bash -c \
'jupyter nbconvert --execute --to figS7-topology.ipynb --inplace figS7-topology.ipynb'
# screen -dmS S8 bash -c \
# 'jupyter nbconvert --execute --to figS8-RW.ipynb --inplace figS8-RW.ipynb'
# screen -dmS S10 bash -c \
# 'jupyter nbconvert --execute --to figS10-rnn.ipynb --inplace figS10-rnn.ipynb'
