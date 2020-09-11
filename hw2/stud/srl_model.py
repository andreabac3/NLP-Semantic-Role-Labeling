# !/usr/bin/env python3

import torch
from torch import nn

from torchcrf import CRF

from stud.biaffine_classifier import BiaffineAttention
from stud.gcn import GCN


def batch_adj_matrix(adj_matrix_list: torch.Tensor) -> torch.Tensor:
    '''
    :param adj_matrix_list: shape (batch_size, side_square, side_square)
    :return: a torch.Tensor with matrix belong his diagonal
    in order to support batches for the GCN it is necessary
    to create a matrix that will contain all the adjacency matrices in a batch along its diagonal.
    as shown: https://user-images.githubusercontent.com/7347296/34198790-eb5bec96-e56b-11e7-90d5-157800e042de.png
    In order to contribute to the open source world, I created a pull request to add this function to the kipf (GCN creator) repository.
    Then I closed the pull request thinking it was better to wait for the end of the exam.
    pull request: https://github.com/tkipf/pygcn/pull/65
    '''
    dimension = adj_matrix_list.shape
    batch_size = dimension[0]
    side_of_the_square = dimension[2]
    side_batch_matrix = side_of_the_square * batch_size
    res_batch_matrix = torch.zeros((side_batch_matrix, side_batch_matrix))
    for batch_num in range(batch_size):
        res_batch_matrix[side_of_the_square * batch_num:side_of_the_square + (batch_num * side_of_the_square), side_of_the_square * batch_num:side_of_the_square + (batch_num * side_of_the_square)] = adj_matrix_list[batch_num]

    return res_batch_matrix


class SRL_final_MODEL(nn.Module):
    def __init__(self, hparams, configurator: dict):
        super(SRL_final_MODEL, self).__init__()

        self.device: str = hparams.device

        self.dropout = nn.Dropout(hparams.dropout)

        self.real_lstm_hidden_dim: int = hparams.lstm_hidden_dim * 2 if hparams.bidirectional else hparams.lstm_hidden_dim

        self.lstm_input_size: int = hparams.embedding_dim_words + hparams.embedding_dim_pos + hparams.embedding_dim_lemmas + hparams.embedding_dim_relations + hparams.embedding_dim_predicates

        self.word_embedding = nn.Embedding(hparams.vocab_size_words, hparams.embedding_dim_words, padding_idx=0)
        self.word_embedding_pos = nn.Embedding(hparams.vocab_size_pos_tags, hparams.embedding_dim_pos, padding_idx=0)
        self.lemma_embedding = nn.Embedding(hparams.vocab_size_lemmas, hparams.embedding_dim_lemmas, padding_idx=0)

        self.dependency_relations_embedding = nn.Embedding(hparams.vocab_size_dependency_relations, hparams.embedding_dim_relations, padding_idx=0)

        self.predicates_embedding = nn.Embedding(hparams.vocab_size_predicates, hparams.embedding_dim_predicates, padding_idx=0)

        self.use_pretrained: bool = configurator['use_pretrained']
        if self.use_pretrained:
            # LOAD GloVe embeddings
            self.word_embedding.weight.data.copy_(hparams.glove_embeddings)

        # --- CRF Layer ---
        self.use_crf: bool = configurator['use_crf']
        if self.use_crf:
            print("we are using crf")
            self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)

        # --- BERT EMB ---
        self.use_bert_embeddings: bool = configurator['use_bert_embeddings']
        if self.use_bert_embeddings:
            self.lstm_bert_emb = nn.LSTM(hparams.bert_hidden_dim, hparams.lstm_hidden_dim,
                                         bidirectional=hparams.bidirectional,
                                         batch_first=True,
                                         num_layers=hparams.bert_lstm_num_layers,
                                         dropout=hparams.lstm_dropout)
            self.lstm_input_size += self.real_lstm_hidden_dim

        # --- GCN Layer ---
        self.use_dependecy_heads: bool = configurator['use_dependecy_heads']
        if self.use_dependecy_heads:
            self.gcn_layer = GCN(nfeat=self.real_lstm_hidden_dim, nhid=hparams.gcn_hidden_dim, nclass=hparams.gcn_output_dim, dropout=hparams.gcn_dropout_probability)

            self.gcn_bilstm = nn.LSTM(self.lstm_input_size, hparams.lstm_hidden_dim,
                                      bidirectional=hparams.bidirectional,
                                      batch_first=True,
                                      num_layers=hparams.gcn_lstm_num_layers,
                                      dropout=hparams.lstm_dropout)
            self.lstm_input_size += hparams.gcn_output_dim

        self.lstm_emb = nn.LSTM(self.lstm_input_size, hparams.lstm_hidden_dim,
                                bidirectional=hparams.bidirectional,
                                batch_first=True,
                                num_layers=hparams.num_layers,
                                dropout=hparams.lstm_dropout)

        self.output_layer = nn.Linear(self.real_lstm_hidden_dim, hparams.num_classes)

        # --- Biaffine Layer ---
        self.use_biaffine_layer = configurator['use_biaffine_layer']
        if self.use_biaffine_layer:
            self.output_layer_2 = nn.Linear(self.real_lstm_hidden_dim, hparams.num_classes)
            self.bilstm_stacked = nn.LSTM(self.real_lstm_hidden_dim, hparams.lstm_hidden_dim,
                                          bidirectional=hparams.bidirectional,
                                          batch_first=True,
                                          num_layers=hparams.biaffine_lstm_num_layers,
                                          dropout=hparams.lstm_dropout)

            self.biaffine_scorer: BiaffineAttention = BiaffineAttention(hparams.num_classes, hparams.num_classes)
        self.use_predicate_biaffine = configurator['use_predicates'] and self.use_biaffine_layer
        if self.use_predicate_biaffine:
            self.biaffine_lstm_predicate = nn.LSTM(hparams.embedding_dim_predicates, hparams.lstm_hidden_dim,
                                                   bidirectional=hparams.bidirectional,
                                                   batch_first=True,
                                                   num_layers=hparams.biaffine_lstm_num_layers,
                                                   dropout=hparams.lstm_dropout)

    def forward(self, sample):
        x_word = sample['words']
        x_pos = sample['pos_tags']
        x_predicate = sample['predicates']
        x_lemma = sample['lemmas']
        x_dependency_relation = sample['dependency_relations']

        dimension: torch.Size = x_word.shape
        batch_size: int = dimension[0]
        sequence_length: int = dimension[1]

        #  Produce the embeddings vectors starting from the encoded features
        word_emb: torch.Tensor = self.word_embedding(x_word)
        pos_emb: torch.Tensor = self.word_embedding_pos(x_pos)
        lemma_emb: torch.Tensor = self.lemma_embedding(x_lemma)
        predicates_emb = self.predicates_embedding(x_predicate)

        relations_emb: torch.Tensor = self.dependency_relations_embedding(x_dependency_relation)
        word_representation: torch.Tensor = torch.cat((word_emb, pos_emb, lemma_emb, relations_emb, predicates_emb), dim=2)  # core word representation

        if self.use_bert_embeddings:
            bert_embeddings = sample['bert_embeddings'].to(self.device)
            lstm_bert, _ = self.lstm_bert_emb(bert_embeddings.to(torch.float))
            bert_embeddings = bert_embeddings.to('cpu').to(torch.float16)  # Used only in gpu mode: move bert emb back to cpu.
            # I store the bert embeddings into float-16bit representation in order to save memory
            word_representation = torch.cat((word_representation, lstm_bert), dim=2)  # I add the bert emb to the core word repr.

        word_representation = self.dropout(word_representation)  # I apply the dropout to the word representation tensor (concatenation of different embeddings)

        if self.use_dependecy_heads:
            x_dependency_heads_matrix = sample['dependency_heads_matrix']
            gcn_bilstm_out, _ = self.gcn_bilstm(word_representation)  # BiLSTM applied to the word representation and used as Feature matrix for GCN, in order to produce a context-aware input for GCN
            matrix_adj = batch_adj_matrix(x_dependency_heads_matrix).to(self.device)  # following the suggestion of the GCN creator, in order to support the batch training I build a matrix with all adj matrices in his diagonal
            rel_emb = gcn_bilstm_out.reshape(batch_size * sequence_length, self.real_lstm_hidden_dim)
            out_gcn = self.gcn_layer(rel_emb, matrix_adj)
            out_gcn = out_gcn.view(batch_size, sequence_length, -1)
            word_representation = torch.cat((word_representation, out_gcn), dim=2)

        lstm_output, _ = self.lstm_emb(word_representation)

        if self.use_biaffine_layer:
            if self.use_predicate_biaffine:
                # Use the encoding of the predicate embeddings in combination with the core word representation as input for the biaffine attention layer
                predicate_embeddings_encoding = self.biaffine_lstm_predicate(predicates_emb)
                return self.biaffine_scorer(self.output_layer(lstm_output), self.output_layer_2(predicate_embeddings_encoding))
            else:
                # Use the deeper encoding for the biaffine attention layer
                stacked_lstm_output, _ = self.bilstm_stacked(lstm_output)
                return self.biaffine_scorer(self.output_layer(lstm_output), self.output_layer_2(stacked_lstm_output))

        return self.output_layer(lstm_output)
