from typing import List
import argparse
import os
import gpustat
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def def_additional_embeddings(s):
    try:
        embedding, size = s.split(':')
        size = int(size)
    except:
        raise argparse.ArgumentTypeError('Additional embeddings should be in format: TagName:EmbeddingSize.')
    return embedding, size


def def_task(s):
    try:
        task, path = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Task should be in format: TaskName:DataPath.')
    return task, path


parser = argparse.ArgumentParser(description='Train Flair NER model')
parser.add_argument('--task', type=def_task, required=True, help='Task and data path')
parser.add_argument('--tag-type', required=True, help='Tag type to train')
parser.add_argument('--init-model', help='Initial OntoNotes model')
parser.add_argument('--direct-projection-weight', type=float, default=0., help='Weight of direct projection pass')
parser.add_argument('--bypass-weight', type=float, default=0., help='Weight of bypass')
parser.add_argument('--freeze', action='store_true', help='Freeze pretrained model')
parser.add_argument('--additional-embeddings', nargs='*', type=def_additional_embeddings, help="Type(s) of additional input tag embeddings")
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
parser.add_argument('--init-lr', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
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
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter


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
print(corpus)

tag_type = args.tag_type
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

additional_tag_dictionaries = []
additional_tag_embeddings = []
if args.additional_embeddings:
    for tag, size in args.additional_embeddings:
        d = corpus.make_tag_dictionary(tag_type=tag)
        additional_tag_dictionaries.append(d)
        additional_tag_embeddings.append(NonStaticWordEmbeddings(size, d, tag))
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

print('Using additional tag embeddings: {}'.format(str(additional_tag_embeddings)))
print('Using {}'.format(args.optimizer))
print('Initial learning rate: {}'.format(args.init_lr))
# initialize sequence tagger
from flair.models import SequenceTagger


if os.path.isdir(args.working_dir) and os.path.isfile(os.path.join(args.working_dir, 'best-model.pt')):
    print('Model exists in the working directory, loading ...')
    tagger: SequenceTagger = SequenceTagger.load_from_file(os.path.join(args.working_dir, 'best-model.pt'), eval=False)
    print('Tag dict: ' + str(tagger.tag_dictionary.idx2item))
    tagger.freeze_model(args.freeze)
else:
    print('Loading initial model from ' + args.init_model)
    tagger: SequenceTagger = SequenceTagger.load_from_file(args.init_model, eval=False)
    print('Previous tag dict: ' + str(tagger.tag_dictionary.idx2item))
    tagger.reset_tag_dict(tag_type, tag_dictionary)
    print('New tag dict: ' + str(tagger.tag_dictionary.idx2item))

    tagger.freeze_model(args.freeze)
    
    assert args.direct_projection_weight > 0 or args.bypass_weight > 0
    if args.bypass_weight > 0:
        tagger.set_bypass(args.bypass_weight)
    if args.direct_projection_weight > 0:
        tagger.set_direct_projection(args.direct_projection_weight)


from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer)

trainer.train(args.working_dir, EvaluationMetric.MICRO_F1_SCORE, learning_rate=args.init_lr, mini_batch_size=32,
              max_epochs=args.num_epochs, anneal_factor=anneal_factor, embeddings_in_memory=False)

plotter = Plotter()
plotter.plot_training_curves('{}/loss.tsv'.format(args.working_dir))
plotter.plot_weights('{}/weights.txt'.format(args.working_dir))
