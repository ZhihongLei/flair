#!/bin/bash

train=/path/to/flair/train_hybrid.py
task=conll_03
tag=ner
path=/path/to/data
w=glove
lr=0.1
hsz=256
epochs=500
lw=0.5

python3 ${train} \
    --task ${task}:${path} \
    --tag-type ${tag} \
    --lm-weight ${lw} \
    --word-embeddings ${w} \
    --flair-embeddings news-forward news-backward \
    --relearn-embeddings \
    --no-crf \
    --init-lr ${lr} \
    --hidden-size ${hsz} \
    --num-epochs ${epochs} \
    --working-dir /paht/to/expt/dir/${task}-${tag}-hybrid-${lw}-no-crf
