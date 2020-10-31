decoder=/path/to/flair/beam_search_decoding.py
task=conll_03
path=/path/to/data
tagger=/path/to/tagger/best-model.pt
lm=/path/to/lm/best-model.pt
lm_weight=0.5

python3 ${decoder} \
    --tagger-model ${tagger} \
    --language-model ${lm} \
    --task ${task}:${path} \
    --lm-weight ${lm_weight}

