# !/usr/bin/env python3

'''
code inspired by notebook on pos tag from the nlp sapienza group
SOURCE CODE: read_dataset function is taken from the sapienza nlp group, from theirs utils.py


normalize: is taken from the kipf repo https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
adjacency_matrix: portion of this function is taken from the kipf repo https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
'''
import json
from collections import Counter
from typing import List

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from tqdm import tqdm
import numpy as np

import scipy.sparse as sp

from stud.bert_embedder import BERTEmbedder


def read_dataset(path: str):
    '''
    SOURCE CODE: Sapienza NLP group, from theirs utils.py file
    '''
    with open(path) as f:
        dataset = json.load(f)

    sentences, labels = {}, {}
    for sentence_id, sentence in dataset.items():
        sentence_id = int(sentence_id)
        sentences[sentence_id] = {
            'words': sentence['words'],
            'lemmas': sentence['lemmas'],
            'pos_tags': sentence['pos_tags'],
            'dependency_heads': [int(head) for head in sentence['dependency_heads']],
            'dependency_relations': sentence['dependency_relations'],
            'predicates': sentence['predicates'],
        }

        labels[sentence_id] = {
            'predicates': sentence['predicates'],
            'roles': {int(p): r for p, r in sentence['roles'].items()}
        }

    return sentences, labels


def build_vocab(dataset, feature_type: str, min_freq: int = 1, pad_token: str = '<pad>', unk_token: str = '<unk>') -> Vocab:
    '''
    build a vocabulary for a given vector
    '''
    counter: Counter = Counter()
    for i in tqdm(range(len(dataset.keys()))):
        list_of_feature = dataset[i][feature_type]
        for word in list_of_feature:
            if word != pad_token:
                counter[word] += 1
    return Vocab(counter, specials=[pad_token, unk_token], min_freq=min_freq)


def build_vocab_roles(dataset, pad_token: str = '<pad>') -> Vocab:
    '''
    build a vocabulary for the roles label vector
    '''
    counter: Counter = Counter()
    for sentence_id in tqdm(range(len(dataset.keys()))):
        roles_dict = dataset[sentence_id]['roles']
        predicate_indices = roles_dict.keys()
        for pred_index in predicate_indices:
            for role in roles_dict[pred_index]:
                if role != pad_token:
                    counter[role] += 1
    return Vocab(counter, specials=[pad_token])


def initialize_matrix(dim, no_arc=0):
    # to avoid cloning list
    return [no_arc] * dim


def adjacency_matrix(tree_list: List[str]):
    '''
    starting from a dependecy heads in which each list element points to his father
    return a adjacency_matrix.
    I add the extra node 0 for the root and the self-loop
    Then I normalized the matrix as suggested in this article https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
    The normalization consist in the multiplication of the adjacency matrix by it's inverse degree matrix

    portion of this function is taken from the kipf repository https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    '''
    tree_list = [0] + [int(elem) for elem in tree_list]  # I add the extra node 0 for the root
    arcs: dict = {i: elem for i, elem in enumerate(tree_list)}  # arcs {1: 0, 2: 2, 3: 3, 0: 0}

    matrix = [initialize_matrix(dim=len(tree_list)) for _ in range(len(tree_list))]
    for i in range(len(matrix)):
        j = arcs[i]
        matrix[i][j] = 1

    for i in range(len(matrix)):
        matrix[0][i] = 0
    A = np.matrix(matrix)
    # start kipf repo utils.py
    A = sp.csr_matrix(A)
    adj = A
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # convert the direct graph to it's indirect version
    adj = normalize(adj + sp.eye(adj.shape[0]))  # add the self loops and then normalize the adjancency matrix
    # end kipf repo utils.py
    return adj.todense()


def normalize(mx):
    '''
    For GCN the normalization of the adj matrix is required to avoid the known problems of vanishing or exploding gradients.
    I normalized the adjacency matrix with indirect edges and with self-loops by its inverse degree matrix.
    SOURCE CODE: https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    '''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class SRL_Dataset(Dataset):

    def __init__(self, sentences: dict, labels: dict = None, device: str = None, pad_token: str = '<pad>', configurator: dict = None, max_len: int = None, bert_model=None, bert_tokenizer=None):
        super(SRL_Dataset, self).__init__()
        assert configurator is not None and device is not None
        self.device: str = device
        self.sentences: dict = sentences
        self.labels: dict = labels
        self.pad_token: str = pad_token
        self.empty_predicates: list = []
        self.max_len: int = max_len if max_len is not None else self._calculate_max_len(sentences=sentences)
        self.configurator: dict = configurator
        if self.configurator['use_bert_embeddings']:
            # bert embedder instantiation
            self.bert_emb: BERTEmbedder = BERTEmbedder(bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=self.device)

    def _calculate_max_len(self, sentences) -> int:
        # calculate the max sentence length in the dataset
        max_len: int = max([len(sentences[id_sentence]['words']) for id_sentence in sentences.keys()])
        return max_len

    def encode_test(self, elem: List[str], vocab: Vocab) -> List[int]:
        # make an encoding of the sentence.
        # from List of string to list of int using the vocabulary
        sample = []
        for i in range(len(elem)):
            if elem[i] not in vocab.stoi:
                sample.append(vocab['<unk>'])
                continue
            sample.append(vocab[elem[i]])
        return sample

    def encode_label(self, labels: List[str], vocab: Vocab) -> List[int]:
        # encode label from class for instance Agent to -> label id
        return [vocab[label] for label in labels]

    def __len__(self) -> int:
        if self.samples is None:
            raise Exception("You should call build_sample()")
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.samples is None:
            raise Exception("You should call build_sample()")
        return self.samples[idx]

    def right_pad_sentence(self, sentence: List[int], pad_token: int = 0) -> List[int]:
        '''
        Takes an encoded sentences and return the same sentence with a fixed pad length
        '''
        padded_sequence: List[int] = [pad_token] * self.max_len
        for i, word in enumerate(sentence):
            padded_sequence[i] = word
        return padded_sequence

    def _to_one_hot(self, enc_predicates: List[int], index: int, vocab_predicates, place_holder: str = '_') -> List[int]:
        '''
        Remove other predicates from a predicates list, I take only one predicates given his index.
        example
        [_,_,_,Agent,_,_,_,Topic], index = 3 --return--> [_,_,_,Agent,_,_,_,_]
        '''
        one_hot_predicates: List[int] = [vocab_predicates[place_holder]] * len(enc_predicates)
        one_hot_predicates[index] = enc_predicates[index]  # insert the right predicate at the given index
        return one_hot_predicates

    def build_sample(self, vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label_roles, vocab_dependency_relations, place_holder: str = '_'):

        samples = []

        for sentence_id in tqdm(self.sentences.keys()):
            words_list = self.sentences[sentence_id]['words']
            len_words_list: int = len(words_list)
            pos_list: List[str] = self.sentences[sentence_id]['pos_tags']
            lemmas_list: List[str] = self.sentences[sentence_id]['lemmas']
            predicates_list: List[str] = self.sentences[sentence_id]['predicates']

            if set(predicates_list) == set('_'):
                # Skip all sentences without predicates and store their id
                self.empty_predicates.append(sentence_id)
                continue

            # -- Non trainable Bert Embeddings --
            if self.configurator['use_bert_embeddings']:
                # Calculate bert embeddings
                # I move the bert embeddings tensor to the CPU, I will move the bert embeddings to the right device during the train to avoid the gpu out of memory
                vector_bert_emb = self.bert_emb.embed_sentences([words_list])[0].to('cpu')
                diff: int = self.max_len - len_words_list
                pad_value: int = 0
                vector_bert_emb: torch.Tensor = torch.nn.functional.pad(vector_bert_emb.T, (0, diff), value=pad_value).T  # pad the bert embeddings
                vector_bert_emb: torch.FloatTensor = vector_bert_emb.to(torch.float16)  # In order to reduce the memory usage I move to the float16 representation

            # encoding of the sentence
            enc_words: List[int] = self.right_pad_sentence(self.encode_test(words_list, vocab_words), pad_token=vocab_words['<pad>'])
            enc_words: torch.LongTensor = torch.LongTensor(enc_words).to(self.device)

            enc_pos: List[int] = self.right_pad_sentence(self.encode_test(pos_list, vocab_pos_tags), pad_token=vocab_pos_tags['<pad>'])
            enc_pos: torch.LongTensor = torch.LongTensor(enc_pos).to(self.device)

            enc_lemmas: List[int] = self.right_pad_sentence(self.encode_test(lemmas_list, vocab_lemmas), pad_token=vocab_lemmas['<pad>'])
            enc_lemmas: torch.LongTensor = torch.LongTensor(enc_lemmas).to(self.device)

            dependency_relations_list: List[str] = self.sentences[sentence_id]['dependency_relations']
            enc_dependency_relations: List[int] = self.right_pad_sentence(self.encode_test(dependency_relations_list, vocab_dependency_relations), pad_token=vocab_dependency_relations['<pad>'])
            enc_dependency_relations: torch.LongTensor = torch.LongTensor(enc_dependency_relations).to(self.device)

            if self.configurator['use_dependecy_heads']:
                # Create adjacency_matrix starting from dependency head (tree), then pad the adj matrix

                dependency_heads_list: List[str] = self.sentences[sentence_id]['dependency_heads']
                # Create the adjacency matrix, the matrix is already normalized
                dependency_heads_list: torch.tensor = torch.tensor(dependency_heads_list, device=self.device)
                dependency_heads_matrix: torch.FloatTensor = torch.FloatTensor(adjacency_matrix(dependency_heads_list)).to(self.device)  # return a adjacency matrix
                diff: int = self.max_len - len(dependency_heads_list) - 1
                pad_value: int = 0
                # Padding of the adjacency matrix
                heads_matrix_pad: torch.Tensor = torch.nn.functional.pad(dependency_heads_matrix, (0, diff), value=pad_value)
                heads_matrix_pad: torch.Tensor = torch.nn.functional.pad(heads_matrix_pad.T, (0, diff), value=pad_value).T
                dependency_heads_matrix = heads_matrix_pad

            enc_predicates: List[int] = self.encode_test(predicates_list, vocab_predicates)
            predicate_indices: List[int] = [index for index, predicate in enumerate(predicates_list) if predicate != place_holder]  # return list of index in which the predicates are not place_holders

            for id_predicates in predicate_indices:
                # there is at least one predicates
                one_hot_predicates: List[int] = self._to_one_hot(enc_predicates, id_predicates, vocab_predicates)  # [_,_,_,Agent,_,_,_,Topic], index = 3 --return--> [_,_,_,Agent,_,_,_,_]
                one_hot_predicates: List[int] = self.right_pad_sentence(one_hot_predicates, pad_token=vocab_predicates['<pad>'])
                one_hot_predicates: torch.LongTensor = torch.LongTensor(one_hot_predicates).to(self.device)

                sample: dict = {'words': enc_words, 'pos_tags': enc_pos, 'predicates': one_hot_predicates, 'lemmas': enc_lemmas,
                                'predicate_id': id_predicates, 'sentence_id': sentence_id, "dependency_relations": enc_dependency_relations
                                }

                if self.configurator['use_bert_embeddings']:
                    sample['bert_embeddings'] = vector_bert_emb
                if self.labels is not None:
                    label_list: List[int] = self.encode_label(self.labels[sentence_id]['roles'][id_predicates], vocab_label_roles)
                    padded_label: List[int] = self.right_pad_sentence(sentence=label_list, pad_token=vocab_label_roles['<pad>'])
                    label_list: torch.LongTensor = torch.LongTensor(padded_label).to(self.device)
                    sample['label'] = label_list
                if self.configurator['use_dependecy_heads']:
                    sample['dependency_heads_matrix'] = dependency_heads_matrix

                samples.append(sample)

        self.samples = samples
        return samples
