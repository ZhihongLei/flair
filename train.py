from typing import List
import argparse
import os
import gpustat
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def pooled_embeddings(s):
    try:
        embedding, pooling = s.split(',')
    except:
        raise argparse.ArgumentTypeError('Pooled Embeddings should be EmbeddingType,PoolingType.')
    if pooling == 'max' or pooling == 'min' or pooling == 'mean' or pooling == 'fade':
        return embedding, pooling
    raise argparse.ArgumentTypeError('PoolingType must be chosen from: max, min, mean or fade.')

parser = argparse.ArgumentParser(description='Train Flair NER model')
parser.add_argument('--word-embeddings', nargs='*', help='Type(s) of word embeddings')
parser.add_argument('--char-embeddings', action='store_true', help='Character embeddings trained on task corpus, Lample 2016')
parser.add_argument('--flair-embeddings', nargs='*', help='Type(s) of Flair embeddings')
parser.add_argument('--pooled-flair-embeddings', nargs='*', type=pooled_embeddings, help="Type(s) of pooled Flair embeddings")
parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--dropout-rate', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
parser.add_argument('--init-lr', type=float, required=True, help='Initial learning rate')
parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
#parser.add_argument('--data-dir', required=True, help='Data dir')
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

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, PooledFlairEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [WordEmbeddings(type) for type in args.word_embeddings]
if args.char_embeddings:
     embedding_types.append(CharacterEmbeddings())
if args.flair_embeddings:
    embedding_types.extend([FlairEmbeddings(type) for type in args.flair_embeddings])
if args.pooled_flair_embeddings:
    embedding_types.extend([PooledFlairEmbeddings(type, pooling) for type, pooling in args.pooled_flair_embeddings])

if len(embedding_types) == 0:
    raise ValueError('Must specify at least one embedding type!')

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, '/u/lei/work/data')
#print('Reading data from {} ...'.format(args.data_dir))
#corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(args.data_dir, {0: 'text', 1: 'ner'}, tag_to_biloes='ner')
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)


embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

if args.optimizer == 'sgd':
    optimizer = SGD
    anneal_factor = 0.5
elif args.optimizer == 'adam':
    optimizer = Adam
    anneal_factor = 0.999
else:
    raise ValueError('Cannot recognize optimizer {}'.format(args.optimizer))

print('Using embeddings: {}'.format(str(embedding_types)))
print('Using {}'.format(args.optimizer))
print('Initial learning rate: {}'.format(args.init_lr))
print('{} hidden layers of size {}'.format(args.num_hidden_layers, args.hidden_size))
print('Dropout rate: {}'.format(args.dropout_rate))
# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True,
                                        rnn_layers=args.num_hidden_layers,
                                        rnn_dropout=args.dropout_rate)

# initialize trainer
from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer)

trainer.train(args.working_dir, EvaluationMetric.MICRO_F1_SCORE, learning_rate=args.init_lr, mini_batch_size=32,
              max_epochs=args.num_epochs, anneal_factor=anneal_factor, test_mode=True)

plotter = Plotter()
plotter.plot_training_curves('{}/loss.tsv'.format(args.working_dir))
plotter.plot_weights('{}/weights.txt'.format(args.working_dir))
