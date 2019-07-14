import argparse
import os
import logging

log = logging.getLogger('flair')

def def_task(s):
    try:
        task, path = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Task should be in format: TaskName:DataPath.')
    return task, path


parser = argparse.ArgumentParser(description='Train Flair model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--tag-type', required=True, help='Tag type to train')
parser.add_argument('--tagger-model', required=True, help='Path to the tagger model')
parser.add_argument('--language-model', required=True, help='Path to the tag language model')
parser.add_argument('--lm-weight', type=float, default=0.2, help='Beam size')
parser.add_argument('--beam-size', type=int, default=10, help='Beam size')
parser.add_argument('--lm-score-type', choices=['log-softmax', 'logits'], default='log-softmax',  help='Type of LM score')

args = parser.parse_args()


from flair.models.language_model import MyLanguageModel, MySimpleLanguageModel
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.models import SequenceTagger
from flair.models.sequence_tagger_model import beam_search, evalute_beam_search


tagger = SequenceTagger.load_from_file(args.tagger_model, eval=True)
lm = MySimpleLanguageModel.load_from_file(args.language_model)


tag_type = args.tag_type
task, path = args.task
beam_size = args.beam_size
log.info('Task {}'.format(task))
log.info('Tag type {}'.format(tag_type))
log.info(f'Beam size {beam_size}')
log.info(f'LM weight: {args.lm_weight}')
log.info(f'LM score type: {args.lm_score_type}')
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)
log.info(corpus)

metric, _ = evalute_beam_search(tagger, lm, corpus.test, args.lm_weight,
                                args.beam_size,
                                emission_score_type='log-softmax' if tagger.use_crf else 'logits',
                                lm_score_type=args.lm_score_type)
print(metric.micro_avg_f_score())