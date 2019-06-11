from typing import List
import argparse
import os
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask, Sentence, Token
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, \
                            FlairEmbeddings, PooledFlairEmbeddings, NonStaticWordEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
from flair.models.language_model import MyLanguageModel
from flair.trainers.language_model_trainer import MyLMTrainer
import logging

log = logging.getLogger('flair')

def def_additional_embeddings(s):
    try:
        tag, size = s.split(':')
        size = int(size)
    except:
        raise argparse.ArgumentTypeError('Additional embeddings should be in format: TagName:EmbeddingSize.')
    return tag, size

def def_task(s):
    try:
        task, path = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Task should be in format: TaskName:DataPath.')
    return task, path



parser = argparse.ArgumentParser(description='Train Flair model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--tag-type', required=True, help='Tag type to train')
parser.add_argument('--embedding-size', type=int, help='Embedding size')
parser.add_argument('--additional-embeddings', nargs='*', type=def_additional_embeddings, help="Type(s) of additional input tag embeddings")
parser.add_argument('--window-size', type=int, default=1, help='Window size of additional embeddings')
parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--dropout-rate', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--working-dir', default='.', help='Working directory where outputs are stored')

args = parser.parse_args()
log.info("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))


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


if os.path.isdir(args.working_dir) and os.path.isfile(os.path.join(args.working_dir, 'best-model.pt')):
    log.info('Loading initial model from ' + os.path.join(args.working_dir, 'best-model.pt'))
    model = MyLanguageModel.load_from_file(os.path.join(args.working_dir, 'best-model.pt'))
    embedding_size = model.embedding_size
    additional_embeddings = model.additional_embeddings
    additional_dictionaries = model.additional_dictionaries
    num_hidden_layers = model.nlayers
    hidden_size = model.hidden_size
else:
    log.info('Initialize model')

    from flair.models import SequenceTagger
    tagger = SequenceTagger.load_from_file('/Users/zhihonglei/work/hiwi/conll03-ner-word-task-trained-256-0.1/best-model.pt', eval=False)
    #dictionary = corpus.make_vocab_dictionary(min_freq=2) if tag_type == 'text' else corpus.make_tag_dictionary(tag_type)
    dictionary = tagger.tag_dictionary
    embedding_size = args.embedding_size
    embeddings = NonStaticWordEmbeddings(embedding_size, dictionary, tag_type)
    
    additional_dictionaries = []
    additional_embeddings = []
    if args.additional_embeddings:
        for tag, size in args.additional_embeddings:
            d = corpus.make_tag_dictionary(tag_type=tag)
            additional_dictionaries.append(d)
            additional_embeddings.append(NonStaticWordEmbeddings(size, d, tag, args.window_size))
    additional_embeddings = torch.nn.ModuleList(additional_embeddings)
    

    num_hidden_layers = args.num_hidden_layers
    hidden_size = args.hidden_size
    
    model = MyLanguageModel(tag_type=tag_type, 
                        embeddings=embeddings, 
                        dictionary=dictionary, 
                        additional_embeddings=additional_embeddings,
                        additional_dictionaries=additional_dictionaries,
                        hidden_size=hidden_size, 
                        nlayers=num_hidden_layers)


log.info('Embedding size: {}'.format(embedding_size))
log.info('Using additional embeddings: {}'.format(str(additional_embeddings)))
log.info('{} hidden layers of size {}'.format(num_hidden_layers, hidden_size))

    
lr = args.lr
working_dir = args.working_dir
if args.optimizer == 'sgd':
    optimizer = SGD
    anneal_factor = 0.5
else:
    optimizer = Adam
    anneal_factor = 0.9999
    

log.info('Using {}'.format(args.optimizer))
log.info('Initial learning rate: {}'.format(lr))
log.info('Dropout rate: {}'.format(args.dropout_rate))
log.info('Working dir: ' + working_dir)    


trainer = MyLMTrainer(model, corpus, optimizer)
trainer.train(base_path=working_dir, learning_rate=lr, mini_batch_size=16, max_epochs=args.num_epochs)