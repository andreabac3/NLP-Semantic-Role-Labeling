# !/usr/bin/env python3

import torch
from typing import List
from transformers import BertModel, BertTokenizer

'''

SOURCE CODE: SAPIENZA NLP GROUP -> NLP Notebook #7 - Sense Embeddings.ipynb

'''


class BERTEmbedder:

    def __init__(self, bert_model: BertModel,
                 bert_tokenizer: BertTokenizer,
                 device: str):
        """
        Args:
          bert_model (BertModel): The pretrained BERT model.
          bert_tokenizer (BertTokenizer): The pretrained BERT tokenizer.
          token_limit (integer): The maximum number of tokens to give as input to BERT.
          device (string): The device on which BERT should run, either cuda or cpu.
        """
        super(BERTEmbedder, self).__init__()
        self.bert_model = bert_model
        self.bert_model.to(device)
        self.bert_model.eval()
        self.bert_tokenizer = bert_tokenizer
        self.device = device

    def embed_sentences(self, sentences: List[str]):
        # we convert the sentences to an input that can be fed to BERT
        input_ids, to_merge_wordpieces, attention_mask, token_type_ids = self._prepare_input(sentences)
        # we set output_all_encoded_layers to True cause we want to sum the
        # representations of the last four hidden layers
        with torch.no_grad():
            # The BertModel forward method returns a tuple of 3 elements:
            # 1) last_hidden_states of shape (batch_size x sequence_length x hidden_size),
            # which is the sequence of hidden states of the last layer of the model,
            # 2) pooler_output of shape batch_size x hidden_size,
            # which is the hidden states of the first token of the sequence (the CLS token)
            # passed through a Linear layer with a Tanh activation function,
            # 3) hidden_states, which is a tuple of FloatTensors, each of shape
            # (batch_size x sequence_length x hidden_size), each FloatTensor is the hidden states
            # of the model at the output of one of BERT's layers.
            bert_output = self.bert_model.forward(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask)

        # we sum the sum of the last four hidden layers (-1 is the hidden states, see point (3) above)
        layers_to_sum = torch.stack([bert_output[-1][x] for x in [-1, -2, -3, -4]], axis=0)
        summed_layers = torch.sum(layers_to_sum, axis=0)
        merged_output = self._merge_embeddings(summed_layers, to_merge_wordpieces)

        return merged_output

    def _prepare_input(self, sentences: List[str]):
        input_ids = []
        # we must keep track of which words have been split so we can merge them afterwards
        to_merge_wordpieces = []
        # BERT requires the attention mask in order to know on which tokens it has to attend to
        # padded indices do not have to be attended to so will be 0
        attention_masks = []
        # BERT requires token type ids for doing sequence classification
        # in our case we do not need them so we set them all to 0
        token_type_ids = []
        # we sum 2 cause we have to consider also [CLS] and [SEP] in the sentence length
        max_len = max([len(self._tokenize_sentence(s)[0]) for s in sentences])
        for sentence in sentences:
            encoded_sentence, to_merge_wordpiece = self._tokenize_sentence(sentence)
            att_mask = [1] * len(encoded_sentence)
            att_mask = att_mask + [0] * (max_len - len(encoded_sentence))
            # we pad sentences shorter than the max length of the batch
            encoded_sentence = encoded_sentence + [0] * (max_len - len(encoded_sentence))
            input_ids.append(encoded_sentence)
            to_merge_wordpieces.append(to_merge_wordpiece)
            attention_masks.append(att_mask)
            token_type_ids.append([0] * len(encoded_sentence))
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_masks = torch.LongTensor(attention_masks).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
        return input_ids, to_merge_wordpieces, attention_masks, token_type_ids

    def _tokenize_sentence(self, sentence: List[str]):
        encoded_sentence = [self.bert_tokenizer.cls_token_id]
        # each sentence must start with the special [CLS] token
        to_merge_wordpiece = []
        # we tokenize a word at the time so we can know which words are split into multiple subtokens
        for word in sentence:
            encoded_word = self.bert_tokenizer.tokenize(word)
            # we take note of the indices associated with the same word
            to_merge_wordpiece.append([i for i in range(len(encoded_sentence) - 1, len(encoded_sentence) + len(encoded_word) - 1)])
            encoded_sentence.extend(self.bert_tokenizer.convert_tokens_to_ids(encoded_word))
        # each sentence must end with the special [SEP] token
        encoded_sentence.append(self.bert_tokenizer.sep_token_id)
        return encoded_sentence, to_merge_wordpiece

    # aggregated_layers has shape: shape batch_size x sequence_length x hidden_size
    def _merge_embeddings(self, aggregated_layers: List[List[float]],
                          to_merge_wordpieces: List[List[int]]):
        merged_output = []
        # first we remove the [CLS] and [SEP] tokens from the output embeddings
        aggregated_layers = aggregated_layers[:, 1:-1, :]
        for embeddings, sentence_to_merge_wordpieces in zip(aggregated_layers, to_merge_wordpieces):
            sentence_output = []
            # for each word we retrieve the indices of its subtokens in the tokenized sentence
            for word_to_merge_wordpiece in sentence_to_merge_wordpieces:
                # we average all the embeddings of the subpieces of a word
                sentence_output.append(torch.mean(embeddings[word_to_merge_wordpiece], axis=0))
            merged_output.append(torch.stack(sentence_output).to(self.device))
        return merged_output
