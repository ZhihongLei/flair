from pathlib import Path

import torch.nn as nn
import torch
import math
from typing import Union, Tuple
from typing import List
import warnings

from torch.optim import Optimizer

import flair
from flair.data import Dictionary, Sentence
import numpy as np
import logging

log = logging.getLogger('flair')


class LanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 dictionary: Dictionary,
                 is_forward_lm: bool,
                 hidden_size: int,
                 nlayers: int,
                 embedding_size: int = 100,
                 nout=None,
                 dropout=0.1):

        super(LanguageModel, self).__init__()

        self.dictionary = dictionary
        self.is_forward_lm: bool = is_forward_lm

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(len(dictionary), embedding_size)

        if nlayers == 1:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)

        self.hidden = None

        self.nout = nout
        if nout is not None:
            self.proj = nn.Linear(hidden_size, nout)
            self.initialize(self.proj.weight)
            self.decoder = nn.Linear(nout, len(dictionary))
        else:
            self.proj = None
            self.decoder = nn.Linear(hidden_size, len(dictionary))

        self.init_weights()

        # auto-spawn on GPU if available
        self.to(flair.device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.detach().uniform_(-initrange, initrange)
        self.decoder.bias.detach().fill_(0)
        self.decoder.weight.detach().uniform_(-initrange, initrange)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, input, hidden, ordered_sequence_lengths=None):
        encoded = self.encoder(input)
        emb = self.drop(encoded)

        self.rnn.flatten_parameters()

        output, hidden = self.rnn(emb, hidden)

        if self.proj is not None:
            output = self.proj(output)

        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).detach()
        return (weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
                weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach())

    def get_representation(self, strings: List[str], chars_per_chunk: int = 512):

        # cut up the input into chunks of max charlength = chunk_size
        longest = len(strings[0])
        chunks = []
        splice_begin = 0
        for splice_end in range(chars_per_chunk, longest, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in strings])
            splice_begin = splice_end

        chunks.append([text[splice_begin:longest] for text in strings])
        hidden = self.init_hidden(len(chunks[0]))

        output_parts = []

        # push each chunk through the RNN language model
        for chunk in chunks:

            sequences_as_char_indices: List[List[int]] = []
            for string in chunk:
                char_indices = [self.dictionary.get_idx_for_item(char) for char in string]
                sequences_as_char_indices.append(char_indices)

            batch = torch.LongTensor(sequences_as_char_indices).transpose(0, 1)
            batch = batch.to(flair.device)

            prediction, rnn_output, hidden = self.forward(batch, hidden)
            rnn_output = rnn_output.detach()

            output_parts.append(rnn_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output

    def get_output(self, text: str):
        char_indices = [self.dictionary.get_idx_for_item(char) for char in text]
        input_vector = torch.LongTensor([char_indices]).transpose(0, 1)

        hidden = self.init_hidden(1)
        prediction, rnn_output, hidden = self.forward(input_vector, hidden)

        return self.repackage_hidden(hidden)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == torch.Tensor:
            return h.clone().detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initialize(self, matrix):
        in_, out_ = matrix.size()
        stdv = math.sqrt(3. / (in_ + out_))
        matrix.detach().uniform_(-stdv, stdv)

    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):

        state = torch.load(str(model_file), map_location=flair.device)

        model = LanguageModel(state['dictionary'],
                              state['is_forward_lm'],
                              state['hidden_size'],
                              state['nlayers'],
                              state['embedding_size'],
                              state['nout'],
                              state['dropout'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(flair.device)

        return model

    @classmethod
    def load_checkpoint(cls, model_file: Path):
        state = torch.load(str(model_file), map_location=flair.device)

        epoch = state['epoch'] if 'epoch' in state else None
        split = state['split'] if 'split' in state else None
        loss = state['loss'] if 'loss' in state else None
        optimizer_state_dict = state['optimizer_state_dict'] if 'optimizer_state_dict' in state else None

        model = LanguageModel(state['dictionary'],
                              state['is_forward_lm'],
                              state['hidden_size'],
                              state['nlayers'],
                              state['embedding_size'],
                              state['nout'],
                              state['dropout'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(flair.device)

        return {'model': model, 'epoch': epoch, 'split': split, 'loss': loss,
                'optimizer_state_dict': optimizer_state_dict}

    def save_checkpoint(self, file: Path, optimizer: Optimizer, epoch: int, split: int, loss: float):
        model_state = {
            'state_dict': self.state_dict(),
            'dictionary': self.dictionary,
            'is_forward_lm': self.is_forward_lm,
            'hidden_size': self.hidden_size,
            'nlayers': self.nlayers,
            'embedding_size': self.embedding_size,
            'nout': self.nout,
            'dropout': self.dropout,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'split': split,
            'loss': loss
        }

        torch.save(model_state, str(file), pickle_protocol=4)

    def save(self, file: Path):
        model_state = {
            'state_dict': self.state_dict(),
            'dictionary': self.dictionary,
            'is_forward_lm': self.is_forward_lm,
            'hidden_size': self.hidden_size,
            'nlayers': self.nlayers,
            'embedding_size': self.embedding_size,
            'nout': self.nout,
            'dropout': self.dropout
        }

        torch.save(model_state, str(file), pickle_protocol=4)

    def generate_text(self, prefix: str = '\n', number_of_characters: int = 1000, temperature: float = 1.0,
                      break_on_suffix=None) -> Tuple[str, float]:

        if prefix == '':
            prefix = '\n'

        with torch.no_grad():
            characters = []

            idx2item = self.dictionary.idx2item

            # initial hidden state
            hidden = self.init_hidden(1)

            if len(prefix) > 1:

                char_tensors = []
                for character in prefix[:-1]:
                    char_tensors.append(
                        torch.tensor(self.dictionary.get_idx_for_item(character)).unsqueeze(0).unsqueeze(0))

                input = torch.cat(char_tensors)
                if torch.cuda.is_available():
                    input = input.cuda()

                prediction, _, hidden = self.forward(input, hidden)

            input = torch.tensor(self.dictionary.get_idx_for_item(prefix[-1])).unsqueeze(0).unsqueeze(0)

            log_prob = 0.

            for i in range(number_of_characters):

                if torch.cuda.is_available():
                    input = input.cuda()

                # get predicted weights
                prediction, _, hidden = self.forward(input, hidden)
                prediction = prediction.squeeze().detach()
                decoder_output = prediction

                # divide by temperature
                prediction = prediction.div(temperature)

                # to prevent overflow problem with small temperature values, substract largest value from all
                # this makes a vector in which the largest value is 0
                max = torch.max(prediction)
                prediction -= max

                # compute word weights with exponential function
                word_weights = prediction.exp().cpu()

                # try sampling multinomial distribution for next character
                try:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                except:
                    word_idx = torch.tensor(0)

                # print(word_idx)
                prob = decoder_output[word_idx]
                log_prob += prob

                input = word_idx.detach().unsqueeze(0).unsqueeze(0)
                word = idx2item[word_idx].decode('UTF-8')
                characters.append(word)

                if break_on_suffix is not None:
                    if ''.join(characters).endswith(break_on_suffix):
                        break

            text = prefix + ''.join(characters)

            log_prob = log_prob.item()
            log_prob /= len(characters)

            if not self.is_forward_lm:
                text = text[::-1]

            return text, log_prob

    def calculate_perplexity(self, text: str) -> float:

        if not self.is_forward_lm:
            text = text[::-1]

        # input ids
        input = torch.tensor([self.dictionary.get_idx_for_item(char) for char in text[:-1]]).unsqueeze(1)
        input = input.to(flair.device)

        # push list of character IDs through model
        hidden = self.init_hidden(1)
        prediction, _, hidden = self.forward(input, hidden)

        # the target is always the next character
        targets = torch.tensor([self.dictionary.get_idx_for_item(char) for char in text[1:]])
        targets = targets.to(flair.device)

        # use cross entropy loss to compare output of forward pass with targets
        cross_entroy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entroy_loss(prediction.view(-1, len(self.dictionary)), targets).item()

        # exponentiate cross-entropy loss to calculate perplexity
        perplexity = math.exp(loss)

        return perplexity



class MyLanguageModel(nn.Module):
    def __init__(self,
                 tag_type,
                 embeddings,
                 dictionary,
                 hidden_size: int,
                 nlayers: int,
                 dropout=0.,
                 additional_embeddings = [],
                 additional_dictionaries = []):

        super(MyLanguageModel, self).__init__()
        
        
        self.embedding_size = embeddings.embedding_length
        for additional_embedding in additional_embeddings:
            self.embedding_size += additional_embedding.embedding_length
        print('lm embedding_size:', self.embedding_size)
        
        if nlayers == 1:
            self.rnn = nn.LSTM(self.embedding_size, hidden_size, nlayers)
        else:
            self.rnn = nn.LSTM(self.embedding_size, hidden_size, nlayers, dropout=dropout)

        self.tag_type = tag_type

        self.embeddings = embeddings
        self.additional_embeddings = additional_embeddings
        self.dictionary = dictionary
        self.additional_dictionaries = additional_dictionaries
        
        
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)
        
        self.linear = nn.Linear(hidden_size, len(dictionary))

        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-0.1, 0.1)
        
        # auto-spawn on GPU if available
        self.to(flair.device)

        
    def init_state(self, bsz):
        weight = next(self.parameters()).detach()
        return (weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
                weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach())
    
        
    def get_embeddings(self, sentences):
        self.embeddings.embed(sentences)
        for additional_embedding in self.additional_embeddings:
            additional_embedding.embed(sentences)
        longest_token_sequence_in_batch = max([len(sentence.tokens) - 1 for sentence in sentences])
        
        sentence_tensor = torch.zeros(len(sentences), longest_token_sequence_in_batch, self.embedding_size,
                                      dtype=torch.float, device=flair.device)        
        for s_id, sentence in enumerate(sentences):
            sentence_tensor[s_id][:len(sentence)-1] = torch.cat([token.get_embedding().unsqueeze(0)
                                                               for token in sentence.tokens[:-1]], 0)            
        sentence_tensor = sentence_tensor.transpose_(0, 1)
        
        return sentence_tensor
    
    
    def get_targets(self, sentences):
        longest_token_sequence_in_batch = max([len(sentence.tokens) - 1 for sentence in sentences])
        targets = torch.ones(len(sentences), longest_token_sequence_in_batch, dtype=torch.long, device=flair.device) * self.dictionary.get_idx_for_item('<pad>')
        
        for s_id, sentence in enumerate(sentences):
            target = torch.tensor([self.dictionary.get_idx_for_item(token.get_tag(self.tag_type).value if self.tag_type != 'text'
                                                       else token.text) for token in sentence.tokens[1:]])
            targets[s_id][:len(sentence)-1] = target
                    
        return targets
    
    
    def forward_step(self, slice_tensor, hx):
        '''
        slice_tensor: (1, bsz, feat)
        '''
        
        rnn_output, hx = self.rnn(slice_tensor, hx)
        feature = self.linear(rnn_output)
        
        return feature, hx
    
        
    def forward(self, sentences: List[Sentence], sort=True, reduce_loss=True, zero_grad=True):
        if zero_grad: self.zero_grad()
        if sort:
            sentences.sort(key=lambda x: len(x), reverse=True)

        sentence_tensor = self.get_embeddings(sentences)
        targets = self.get_targets(sentences)
        lengths: List[int] = [len(sentence.tokens) - 1 for sentence in sentences]
        
        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.dropout > 0.0:
            sentence_tensor = self.drop(sentence_tensor)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

        rnn_output, hidden = self.rnn(packed)
        
        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)
        
        features = self.linear(sentence_tensor)
        features = features.transpose_(0, 1)

        cross_entroy_loss = torch.nn.CrossEntropyLoss(ignore_index=self.dictionary.get_idx_for_item('<pad>'), reduction='sum' if reduce_loss else 'none')
        loss = cross_entroy_loss(features.transpose_(1, 2), targets)
        #print(lengths[-2], loss[-2])
        num_words = torch.tensor(np.sum(lengths))
        #loss = torch.sum(loss)
        
        return loss/num_words if reduce_loss else loss, num_words
    
    @staticmethod
    def save_torch_model(model_state: dict, model_file: str, pickle_module: str = 'pickle', pickle_protocol: int = 4):
        if pickle_module == 'dill':
            try:
                import dill
                torch.save(model_state, str(model_file), pickle_module=dill)
            except:
                log.warning('-' * 100)
                log.warning('ATTENTION! The library "dill" is not installed!')
                log.warning('Please first install "dill" with "pip install dill" to save the model!')
                log.warning('-' * 100)
                pass
        else:
            torch.save(model_state, str(model_file), pickle_protocol=pickle_protocol)


    def save(self, model_file: Union[str, Path]):
        model_state = {
            'state_dict': self.state_dict(),
            'tag_type': self.tag_type,
            'embeddings': self.embeddings,
            'dictionary': self.dictionary,
            'additional_embeddings': self.additional_embeddings,
            'additional_dictionaries': self.additional_dictionaries,
            'hidden_size': self.hidden_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout
        }
        self.save_torch_model(model_state, str(model_file))

    
    def save_checkpoint(self, model_file: Union[str, Path], optimizer_state: dict, scheduler_state: dict, epoch: int,
                        loss: float):
        model_state = {
            'state_dict': self.state_dict(),
            'tag_type': self.tag_type,
            'embeddings': self.embeddings,
            'dictionary': self.dictionary,
            'additional_embeddings': self.additional_embeddings,
            'additional_dictionaries': self.additional_dictionaries,
            'hidden_size': self.hidden_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }
        self.save_torch_model(model_state, str(model_file))
    
    
    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        state = cls._load_state(model_file)
        model = cls(state['tag_type'],
                            state['embeddings'],
                            state['dictionary'],
                            state['hidden_size'],
                            state['nlayers'],
                            state['dropout'],
                            state['additional_embeddings'],
                            state['additional_dictionaries'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(flair.device)
        return model
        
        
        
    @classmethod
    def load_checkpoint(cls, model_file: Union[str, Path]):
        state = cls._load_state(model_file)
        model = cls.load_from_file(model_file)

        epoch = state['epoch'] if 'epoch' in state else None
        loss = state['loss'] if 'loss' in state else None
        optimizer_state_dict = state['optimizer_state_dict'] if 'optimizer_state_dict' in state else None
        scheduler_state_dict = state['scheduler_state_dict'] if 'scheduler_state_dict' in state else None

        return {
            'model': model, 'epoch': epoch, 'loss': loss,
            'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict
        }
        
    
    @classmethod
    def _load_state(cls, model_file: Union[str, Path]):
        # ATTENTION: suppressing torch serialization warnings. This needs to be taken out once we sort out recursive
        # serialization of torch objects
        # https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = flair.file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location=flair.device)
            return state
        


class MySimpleLanguageModel(nn.Module):
    def __init__(self,
                 tag_type,
                 embedding_size,
                 dictionary,
                 hidden_size: int,
                 nlayers: int,
                 dropout=0.):

        super(MySimpleLanguageModel, self).__init__()

        self.tag_type = tag_type
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.nlayers = nlayers
        self.dictionary = dictionary

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(len(dictionary.item2idx), embedding_size)

        if nlayers == 1:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)

        self.hidden = None

        self.decoder = nn.Linear(hidden_size, len(dictionary))

        #self.init_weights()

        # auto-spawn on GPU if available
        self.to(flair.device)


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.detach().uniform_(-initrange, initrange)
        self.decoder.bias.detach().fill_(0)
        self.decoder.weight.detach().uniform_(-initrange, initrange)



    def forward(self, batch, lengths, reduce_loss=True, zero_grad=True):
        if zero_grad: self.zero_grad()
        batch = batch.to(device=flair.device)
        batch.transpose_(0, 1)
        inputs = batch[:-1].transpose_(0, 1)
        targets = batch[1:].transpose_(0, 1)

        sentence_tensor = self.encoder(inputs)
        sentence_tensor = sentence_tensor.transpose_(0, 1)
        if self.dropout > 0:
            sentence_tensor = self.drop(sentence_tensor)
        #self.rnn.flatten_parameters()
        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)
        rnn_output, hidden = self.rnn(packed)
        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

        if self.dropout > 0:
            sentence_tensor = self.drop(sentence_tensor)

        features = self.decoder(sentence_tensor)
        features = features.transpose_(0, 1)

        cross_entroy_loss = torch.nn.CrossEntropyLoss(ignore_index=self.dictionary.get_idx_for_item('<pad>'),
                                                      reduction='sum' if reduce_loss else 'none')
        loss = cross_entroy_loss(torch.transpose(features, 1, 2), targets)
        num_words = torch.tensor(np.sum(lengths))

        return loss / num_words if reduce_loss else loss, num_words, features


    def forward_step(self, slice, hx):
        slice = slice.to(device=flair.device)
        slice_tensor = self.encoder(slice)
        slice_tensor = slice_tensor.transpose_(0, 1)
        if self.dropout > 0:
            slice_tensor = self.drop(slice_tensor)
        rnn_output, hx = self.rnn(slice_tensor, hx)
        if self.dropout > 0:
            rnn_output = self.drop(rnn_output)
        feature = self.decoder(rnn_output)
        return feature, hx


    def init_state(self, bsz):
        weight = next(self.parameters()).detach()
        return (weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
                weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach())


    def get_word_indices(self, sentences):
        # return [([self.dictionary.get_idx_for_item('<START>')] +
        #          [self.dictionary.get_idx_for_item(token.get_tag(self.tag_type).value) for token in sentence.tokens]) for sentence in sentences]
        return [([self.dictionary.get_idx_for_item('<START>')] +
            [self.dictionary.get_idx_for_item(token.get_tag(self.tag_type).value) for token in sentence.tokens] +
            [self.dictionary.get_idx_for_item('<STOP>')]) for sentence in sentences]


    def get_word_indices_tensor(self, word_indices):
        batch_size, max_seq_len = len(word_indices), max([len(x) for x in word_indices])
        lengths = [len(s) - 1 for s in word_indices]
        batch_data = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=flair.device).fill_(self.dictionary.get_idx_for_item('<pad>'))
        for i in range(batch_size):
            batch_data[i][:lengths[i] + 1] = torch.tensor(word_indices[i], dtype=torch.long, device=flair.device)
        return batch_data, lengths


    @staticmethod
    def save_torch_model(model_state: dict, model_file: str, pickle_module: str = 'pickle', pickle_protocol: int = 4):
        if pickle_module == 'dill':
            try:
                import dill
                torch.save(model_state, str(model_file), pickle_module=dill)
            except:
                log.warning('-' * 100)
                log.warning('ATTENTION! The library "dill" is not installed!')
                log.warning('Please first install "dill" with "pip install dill" to save the model!')
                log.warning('-' * 100)
                pass
        else:
            torch.save(model_state, str(model_file), pickle_protocol=pickle_protocol)

    def save(self, model_file: Union[str, Path]):
        model_state = {
            'state_dict': self.state_dict(),
            'tag_type': self.tag_type,
            'embedding_size': self.embedding_size,
            'dictionary': self.dictionary,
            'hidden_size': self.hidden_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout
        }
        self.save_torch_model(model_state, str(model_file))

    def save_checkpoint(self, model_file: Union[str, Path], optimizer_state: dict, scheduler_state: dict, epoch: int,
                        loss: float):
        model_state = {
            'state_dict': self.state_dict(),
            'tag_type': self.tag_type,
            'embedding_size': self.embedding_size,
            'dictionary': self.dictionary,
            'hidden_size': self.hidden_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }
        self.save_torch_model(model_state, str(model_file))

    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        state = cls._load_state(model_file)
        model = cls(state['tag_type'],
                    state['embedding_size'],
                    state['dictionary'],
                    state['hidden_size'],
                    state['nlayers'],
                    state['dropout'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(flair.device)
        return model

    @classmethod
    def load_checkpoint(cls, model_file: Union[str, Path]):
        state = cls._load_state(model_file)
        model = cls.load_from_file(model_file)

        epoch = state['epoch'] if 'epoch' in state else None
        loss = state['loss'] if 'loss' in state else None
        optimizer_state_dict = state['optimizer_state_dict'] if 'optimizer_state_dict' in state else None
        scheduler_state_dict = state['scheduler_state_dict'] if 'scheduler_state_dict' in state else None

        return {
            'model': model, 'epoch': epoch, 'loss': loss,
            'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict
        }

    @classmethod
    def _load_state(cls, model_file: Union[str, Path]):
        # ATTENTION: suppressing torch serialization warnings. This needs to be taken out once we sort out recursive
        # serialization of torch objects
        # https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = flair.file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location=flair.device)
            return state