import os
import argparse
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus, Sentence, Token
from flair.models.language_model import MyLanguageModel
from flair.trainers.language_model_trainer import MyLMTrainer
import logging

log = logging.getLogger('flair')


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

print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))




tag_type = args.tag_type
task, path = args.task
log.info('Task {}'.format(task))
log.info('Tag type {}'.format(tag_type))
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)
for data in [corpus.train, corpus.dev, corpus.test]:
    for sentence in data:
        start_token = Token('<START>')
        end_token = Token('<STOP>')
        for tag in sentence[0].tags.keys():
            start_token.add_tag(tag, '<START>')
            end_token.add_tag(tag, '<STOP>')
        start_token.idx = -1
        start_token.sentence = sentence
        sentence.tokens.insert(0, start_token)
        sentence.add_token(end_token)
        
log.info(corpus)

log.info('Loading initial model from ' + os.path.join(args.working_dir, 'best-model.pt'))
model = MyLanguageModel.load_from_file(os.path.join(args.working_dir, 'best-model.pt'))

_, final_ppl = MyLMTrainer.evaluate(model, corpus.test, 32)
log.info(f'Test PPL: {final_ppl}')
