#!/bin/bash
train=/path/to/flair/train.py
task=conll_03
tag=ner
path=/path/to/data
w=glove
lr=0.1
hsz=256
epochs=500


python3 ${train} \
    --task ${task}:${path} \
    --tag-type ${tag} \
    --word-embeddings ${w} \
    --pooled-flair-embeddings news-forward:min news-backward:min \
    --relearn-embeddings \
    --init-lr ${lr} \
    --hidden-size ${hsz} \
    --num-epochs ${epochs} \
    --working-dir /path/to/expt/dir/${task}-${tag}
