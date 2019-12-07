import os
import argparse
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus, Sentence, Token
from flair.models.language_model import MyLanguageMode
from flair.trainers.language_model_trainer import MyLMTrainer
import logging

log = logging.getLogger('flair')


def def_task(s):
    try:
        task, path = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Task should be in format: TaskName:DataPath.')
    return task, path


parser = argparse.ArgumentParser(description='Test language model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--working-dir', default='.', help='Working directory where outputs are stored')
args = parser.parse_args()

task, path = args.task
log.info('Task {}'.format(task))
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)

log.info('Loading initial model from ' + os.path.join(args.working_dir, 'best-model.pt'))
model = MyLanguageMode.load_from_file(os.path.join(args.working_dir, 'best-model.pt'))
train_data, dev_data, test_data = [model.get_word_indices(data) for data in [corpus.train, corpus.dev, corpus.test]]
batch_size = args.batch_size
log.info(f'Batch size {batch_size}')

_, final_ppl = MyLMTrainer.evaluate(model, test_data, batch_size)
log.info(f'Test PPL: {final_ppl}')
