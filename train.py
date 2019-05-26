from typing import List
import argparse
import os
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def def_pooled_embeddings(s):
    try:
        embedding, pooling = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Pooled Embeddings should be in format: EmbeddingType:PoolingType.')
    if pooling == 'max' or pooling == 'min' or pooling == 'mean' or pooling == 'fade':
        return embedding, pooling
    raise argparse.ArgumentTypeError('PoolingType must be chosen from: max, min, mean or fade.')


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

def def_additional_model_inputs(s):
    try:
        path, from_type, to_type = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Additional Model Inputs should be in format: ModelPath:FromType:ToType.')
    from_types = ['embeddings', 'hidden', 'logits']
    to_types = ['embeddings', 'hidden']
    if from_type in from_types and to_type in to_types:
        return path, from_type, to_type
    if from_type not in from_types:
        raise argparse.ArgumentTypeError('FromType must be chosen from: {}.'.format(', '.join(from_types)))
    else:
        raise argparse.ArgumentTypeError('ToType must be chosen from: {}.'.format(', '.join(to_types)))


parser = argparse.ArgumentParser(description='Train Flair model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--tag-type', required=True, help='Tag type to train')
parser.add_argument('--word-embeddings', nargs='*', help='Type(s) of word embeddings')
parser.add_argument('--char-embeddings', action='store_true', help='Character embeddings trained on task corpus, Lample 2016')
parser.add_argument('--flair-embeddings', nargs='*', help='Type(s) of Flair embeddings')
parser.add_argument('--relearn-embeddings', action='store_true', help='Re-learn embeddings, might be useful when using pretrained embeddings')
parser.add_argument('--pooled-flair-embeddings', nargs='*', type=def_pooled_embeddings, help="Type(s) of pooled Flair embeddings")
parser.add_argument('--additional-embeddings', nargs='*', type=def_additional_embeddings, help="Type(s) of additional input tag embeddings")
parser.add_argument('--window-size', type=int, default=1, help='Window size of additional embeddings')
parser.add_argument('--additional-model-inputs', nargs='*',type=def_additional_model_inputs, help='Use these models\' output of a given layer as input to some layer of the model')
parser.add_argument('--train-additional-models', action='store_true', help='Also train additional models')
parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--dropout-rate', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--no-crf', action='store_false', help='Do not use CRF')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
parser.add_argument('--init-lr', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--working-dir', default='.', help='Working directory where outputs are stored')


args = parser.parse_args()
print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, \
                            FlairEmbeddings, PooledFlairEmbeddings, NonStaticWordEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter


task_name, path = args.task
if task_name == 'conll03':
    task = NLPTask.CONLL_03
    embeddings_in_memory = True
elif task_name == 'ontoner':
    task = NLPTask.ONTONER
    embeddings_in_memory = False
elif task_name == 'mono':
    task = NLPTask.MONO
    embeddings_in_memory = False
else:
    raise NotImplementedError('{} is not implemented yet'.format(task_name))
print('Task {}'.format(task.value))

corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(task, path)
print(corpus)
tag_type = args.tag_type

print('Corpus has been read')
# initialize embeddings
embedding_types: List[TokenEmbeddings] = [WordEmbeddings(type) if type!='task-trained' \
                                            else NonStaticWordEmbeddings(100, corpus.make_vocab_dictionary(min_freq=2)) \
                                          for type in args.word_embeddings]
if args.char_embeddings:
     embedding_types.append(CharacterEmbeddings())
if args.flair_embeddings:
    embedding_types.extend([FlairEmbeddings(type) for type in args.flair_embeddings])
if args.pooled_flair_embeddings:
    embedding_types.extend([PooledFlairEmbeddings(type, pooling) for type, pooling in args.pooled_flair_embeddings])

if len(embedding_types) == 0:
    raise ValueError('Must specify at least one embedding type!')

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)



tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
assert tag_type == 'ner'

pos_dictionary = corpus.make_tag_dictionary(tag_type='pos')
ner_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

for pos_tag in pos_dictionary.item2idx:
    if pos_tag not in tag_dictionary:
        tag_dictionary.add_item(pos_tag)

# Let's change NER label from 'O' to PoS label
for data in [corpus.train, corpus.dev]:
    for sentence in data:
        for token in sentence:
            if token.get_tag(tag_type).value == 'O':
                token.add_tag_label(tag_type, token.get_tag('pos'))


print(tag_dictionary.idx2item)


additional_tag_dictionaries = []
additional_tag_embeddings = []
if args.additional_embeddings:
    for tag, size in args.additional_embeddings:
        d = corpus.make_tag_dictionary(tag_type=tag)
        additional_tag_dictionaries.append(d)
        additional_tag_embeddings.append(NonStaticWordEmbeddings(size, d, tag, args.window_size))
    additional_tag_embeddings = torch.nn.ModuleList(additional_tag_embeddings)
    for d in additional_tag_dictionaries: print(d.idx2item)

                

if args.optimizer == 'sgd':
    optimizer = SGD
    anneal_factor = 0.5
elif args.optimizer == 'adam':
    optimizer = Adam
    anneal_factor = 0.999
else:
    raise ValueError('Cannot recognize optimizer {}'.format(args.optimizer))

print('Using word embeddings: {}'.format(str(embedding_types)))
print('Using additional tag embeddings: {}'.format(str(additional_tag_embeddings)))
print('Re-learning embeddings: {}'.format(args.relearn_embeddings))
print('Using {}'.format(args.optimizer))
print('Initial learning rate: {}'.format(args.init_lr))
print('{} hidden layers of size {}'.format(args.num_hidden_layers, args.hidden_size))
print('Dropout rate: {}'.format(args.dropout_rate))
print('Using CRF: {}'.format(not args.no_crf))
# initialize sequence tagger
from flair.models import SequenceTagger

additional_model_paths: List[str] = []
additional_model_from_types: List[str] = []
additional_model_to_types: List[str] = []

if args.additional_model_inputs:
    for path, from_type, to_type in args.additional_model_inputs:
        additional_model_paths.append(path)
        additional_model_from_types.append(from_type)
        additional_model_to_types.append(to_type)


if os.path.isdir(args.working_dir) and os.path.isfile(os.path.join(args.working_dir, 'best-model.pt')):
    print('Loading initial model from ' + os.path.join(args.working_dir, 'best-model.pt'))
    tagger: SequenceTagger = SequenceTagger.load_from_file(os.path.join(args.working_dir, 'best-model.pt'), eval=False)
else:
    print('Initialize model')

    tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        ner_dictionary=ner_dictionary,
                                        additional_tag_embeddings=additional_tag_embeddings,
                                        additional_tag_dictionaries=additional_tag_dictionaries,
                                        tag_type=tag_type,
                                        use_crf=not args.no_crf,
                                        rnn_layers=args.num_hidden_layers,
                                        rnn_dropout=args.dropout_rate,
                                        relearn_embeddings=args.relearn_embeddings,
                                        additional_model_paths=additional_model_paths, 
                                        additional_model_from_types=additional_model_from_types, 
                                        additional_model_to_types=additional_model_to_types,
                                        train_additional_models=args.train_additional_models)


print(tagger.parameters)
#print(tagger.state_dict)
from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer)

trainer.train(args.working_dir, EvaluationMetric.MICRO_F1_SCORE, learning_rate=args.init_lr, mini_batch_size=32,
              max_epochs=args.num_epochs, anneal_factor=anneal_factor, embeddings_in_memory=embeddings_in_memory)

plotter = Plotter()
plotter.plot_training_curves('{}/loss.tsv'.format(args.working_dir))
plotter.plot_weights('{}/weights.txt'.format(args.working_dir))
