import gpustat
import os
import argparse
from pathlib import Path
from flair.models.sequence_tagger_model import HybridSequenceTagger


def def_task(s):
    try:
        task, path = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Task should be in format: TaskName:DataPath.')
    return task, path

parser = argparse.ArgumentParser(description='Train Flair NER model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--working-dir', default='.', help='Working directory where outputs are stored')
args = parser.parse_args()

print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))

working_dir = args.working_dir


from torch.optim.sgd import SGD
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus
from flair.training_utils import EvaluationMetric
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger


task, path = args.task
print('Task {}'.format(task))
embeddings_in_memory = True if task == 'conll_03' else False
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)

tagger: SequenceTagger = HybridSequenceTagger.load_from_file(os.path.join(working_dir, 'best-model.pt'))
trainer: ModelTrainer = ModelTrainer(tagger, corpus, SGD)
trainer.final_test(Path(working_dir), embeddings_in_memory, EvaluationMetric.MICRO_F1_SCORE, 32)
