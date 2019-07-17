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
parser.add_argument('--tagger-model', required=True, help='Path to the tagger model')
parser.add_argument('--language-model', required=True, help='Path to the tag language model')
parser.add_argument('--lm-weight', type=float, default=0.2, help='Beam size')
parser.add_argument('--beam-size', type=int, default=-1, help='Beam size')
parser.add_argument('--interpolate', type=bool, default=True, help='Interpolate CRF and RNN Tag LM scores')

args = parser.parse_args()


from flair.models.language_model import MyLanguageModel, MySimpleLanguageModel
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.models import SequenceTagger
from flair.models.sequence_tagger_model import beam_search, evalute_beam_search


tagger = SequenceTagger.load_from_file(args.tagger_model, eval=True)
lm = MySimpleLanguageModel.load_from_file(args.language_model)


task, path = args.task
beam_size = len(tagger.tag_dictionary.item2idx) if args.beam_size == -1 else args.beam_size
log.info(f'Beam size {beam_size}')
log.info(f'LM weight: {args.lm_weight}')
if tagger.use_crf:
    log.info(f'Interpolate CRF and RNN Tag LM scores: {args.interpolate}')
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)
log.info(corpus)

metric, _ = evalute_beam_search(tagger, lm, corpus.test, args.lm_weight, beam_size, args.interpolate)
log.info('F1 score: ' + str(metric.micro_avg_f_score()))