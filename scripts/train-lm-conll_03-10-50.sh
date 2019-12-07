train_lm=/path/to/flair/train_lm.py
task=conll_03
tag=ner
path=/path/to/data
esz=10
hsz=50
lr=2.
epochs=500

python3 ${train_lm} \
    --task ${task}:${path} \
    --tag-type ${tag} \
    --embedding-size ${esz} \
    --hidden-size ${hsz} \
    --lr ${lr} \
    --num-epochs ${epochs} \
    --working-dir /path/to/expt/dir/lm-${task}-${esz}-${hsz}
