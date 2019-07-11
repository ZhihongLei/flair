import argparse
import os
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus
from flair.models.language_model import MySimpleLanguageModel
from flair.trainers.language_model_trainer import MySimpleLMTrainer
import logging

log = logging.getLogger('flair')

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
log.info(corpus)


if os.path.isdir(args.working_dir) and os.path.isfile(os.path.join(args.working_dir, 'best-model.pt')):
    log.info('Loading initial model from ' + os.path.join(args.working_dir, 'best-model.pt'))
    model = MySimpleLanguageModel.load_from_file(os.path.join(args.working_dir, 'best-model.pt'))
    dictionary = model.dictionary
    embedding_size = model.embedding_size
    num_hidden_layers = model.nlayers
    hidden_size = model.hidden_size
    dropout_rate = model.dropout
else:
    log.info('Initialize model')

    from flair.models import SequenceTagger
    #tagger = SequenceTagger.load_from_file('/Users/zhihonglei/work/hiwi/conll03-ner-word-task-trained-256-0.1/best-model.pt', eval=False)
    #dictionary = tagger.tag_dictionary

    dictionary = corpus.make_vocab_dictionary(min_freq=2) if tag_type == 'text' else corpus.make_tag_dictionary(tag_type)
    print(dictionary.item2idx)
    
    embedding_size = args.embedding_size
    num_hidden_layers = args.num_hidden_layers
    hidden_size = args.hidden_size
    dropout_rate = args.dropout_rate
    
    model = MySimpleLanguageModel(tag_type=tag_type,
                        embedding_size=embedding_size,
                        dictionary=dictionary, 
                        dropout=dropout_rate,
                        hidden_size=hidden_size, 
                        nlayers=num_hidden_layers)


log.info('Embedding size: {}'.format(embedding_size))
log.info('{} hidden layers of size {}'.format(num_hidden_layers, hidden_size))
log.info('Dropout rate: {}'.format(dropout_rate))


train_data, dev_data, test_data = [model.get_word_indices(data) for data in [corpus.train, corpus.dev, corpus.test]]

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
log.info('Working dir: ' + working_dir)


trainer = MySimpleLMTrainer(model, train_data, dev_data, test_data, optimizer)
trainer.train(base_path=working_dir, learning_rate=lr, mini_batch_size=16, max_epochs=args.num_epochs, anneal_factor=anneal_factor, anneal_against_train_loss=True)