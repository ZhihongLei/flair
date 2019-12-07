import os
import argparse
from pathlib import Path
from flair.models.sequence_tagger_model import HybridSequenceTagger
import logging

log = logging.getLogger('flair')


def def_task(s):
    try:
        task, path = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Task should be in format: TaskName:DataPath.')
    return task, path


parser = argparse.ArgumentParser(description='Test hybrid NER-LM model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--working-dir', default='.', help='Working directory where outputs are stored')
args = parser.parse_args()

working_dir = args.working_dir


from torch.optim.sgd import SGD
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus
from flair.training_utils import EvaluationMetric
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger


task, path = args.task
print('Task {}'.format(task))
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)
batch_size = args.batch_size
log.info(f'Batch size {batch_size}')


tagger: SequenceTagger = HybridSequenceTagger.load_from_file(os.path.join(working_dir, 'best-model.pt'))
trainer: ModelTrainer = ModelTrainer(tagger, corpus, SGD)
trainer.final_test(Path(working_dir), True, EvaluationMetric.MICRO_F1_SCORE, batch_size)
