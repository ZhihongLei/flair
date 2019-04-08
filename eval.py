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
parser.add_argument('--tag-type', required=True, help='Tag type to train')
parser.add_argument('--working-dir', default='.', help='Working directory where outputs are stored')
args = parser.parse_args()

try:
    gpu_id = -1
    for gpu in gpustat.new_query().gpus:
        if len(gpu.processes) == 0: 
            gpu_id = gpu.index
            break
    assert gpu_id != -1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("Using GPU {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
except:
    print("Using CPU")

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


tag_type = args.tag_type
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
tagger: SequenceTagger = SequenceTagger.load_from_file(os.path.join(working_dir, 'best-model.pt'))


trainer: ModelTrainer = ModelTrainer(tagger, corpus, SGD)
trainer.final_test(Path(working_dir), True, EvaluationMetric.MICRO_F1_SCORE, 32)
