import gpustat
import os
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='Train Flair NER model')
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


corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, '/Users/zhihonglei/work/hiwi')

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
tagger: SequenceTagger = SequenceTagger.load_from_file(os.path.join(working_dir, 'best-model.pt'))


trainer: ModelTrainer = ModelTrainer(tagger, corpus, SGD)
trainer.final_test(Path(working_dir), True, EvaluationMetric.MICRO_F1_SCORE, 32)
