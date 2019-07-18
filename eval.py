import gpustat
import os
import argparse
from pathlib import Path


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
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.training_utils import EvaluationMetric
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger


task_name, path = args.task
if task_name == 'conll03':
    task = NLPTask.CONLL_03
    embeddings_in_memory = True
elif task_name == 'ontoner':
    task = NLPTask.ONTONER
    embeddings_in_memory = False
else:
    raise NotImplementedError('{} is not implemented yet'.format(task_name))
print('Task {}'.format(task.value))
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)


tagger: SequenceTagger = SequenceTagger.load_from_file(os.path.join(working_dir, 'best-model.pt'))
trainer: ModelTrainer = ModelTrainer(tagger, corpus, SGD)
trainer.final_test(Path(working_dir), True, EvaluationMetric.MICRO_F1_SCORE, 32)
