import json
import random
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from model import Model
from stud.constant import *
from stud.srl_model import SRL_final_MODEL
import stud.srl_utility
from stud.srl_dataset import SRL_Dataset
from transformers import BertTokenizer, BertModel, BertConfig

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model_34(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    #  return Baseline()

    return StudentModel(device)


def build_model_234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


def build_model_1234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, return_predicates=False):
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence['pos_tags']:
            prob = self.baselines['predicate_identification'][pos]['positive'] / self.baselines['predicate_identification'][pos]['total']
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(zip(sentence['lemmas'], predicate_identification)):
            if not is_predicate or lemma not in self.baselines['predicate_disambiguation']:
                predicate_disambiguation.append('_')
            else:
                predicate_disambiguation.append(self.baselines['predicate_disambiguation'][lemma])
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence['dependency_relations']:
            prob = self.baselines['argument_identification'][dependency_relation]['positive'] / self.baselines['argument_identification'][dependency_relation]['total']
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(sentence['dependency_relations'], argument_identification):
            if not is_argument:
                argument_classification.append('_')
            else:
                argument_classification.append(self.baselines['argument_classification'][dependency_relation])

        if self.return_predicates:
            return {
                'predicates': predicate_disambiguation,
                'roles': {i: argument_classification for i in predicate_indices},
            }
        else:
            return {'roles': {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path='data/baselines.json'):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device: str):
        self.device: str = device

        store_dataset_train = torch.load("model/dict_vocabs.pth", map_location=self.device)
        self.vocab_words = store_dataset_train["vocab_words"]
        self.vocab_pos_tags = store_dataset_train["vocab_pos_tags"]
        self.vocab_lemmas = store_dataset_train["vocab_lemmas"]
        self.vocab_predicates = store_dataset_train["vocab_predicates"]
        self.vocab_dependency_relations = store_dataset_train["vocab_dependency_relations"]
        self.vocab_label = store_dataset_train["vocab_label"]
        net_configuration: dict = net_configurator(use_bert_embeddings=USE_BERT_EMBEDDINGS, use_crf=USE_CRF, use_biaffine_layer=USE_BIAFFINE_LAYER, use_pretrained=False, use_dependecy_heads=USE_DEPENDENCY_HEADS, use_predicates=False, use_syntagnet=USE_SYNTAGNET)

        # -- BERT --
        self.model_name: str = 'bert-base-cased'
        self.bert_config = BertConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name, config=self.bert_config)

        # Hyperparameters class
        @dataclass
        class HParams:
            label_vocabulary = self.vocab_label
            vocab_size_words: int = len(self.vocab_words)
            lstm_hidden_dim: int = 300
            embedding_dim_words: int = 300
            embedding_dim_lemmas: int = 300
            embedding_dim_relations: int = 300
            embedding_dim_predicates: int = 400
            embedding_dim_pos: int = 300
            gcn_output_dim: int = 143
            gcn_dropout_probability: float = 0.5
            gcn_hidden_dim: int = 250
            gcn_lstm_num_layers: int = 2
            bert_lstm_num_layers: int = 2
            bert_hidden_dim: int = self.bert_config.hidden_size
            num_classes: int = len(self.vocab_label)
            biaffine_lstm_num_layers: int = 2
            bidirectional: bool = True
            num_layers: int = 2
            dropout: float = 0.3
            lstm_dropout: float = 0.3
            vocab_size_pos_tags: int = len(self.vocab_pos_tags)
            vocab_size_lemmas: int = len(self.vocab_lemmas)
            vocab_size_dependency_relations: int = len(self.vocab_dependency_relations)
            vocab_size_predicates: int = len(self.vocab_predicates)
            device: str = self.device

        hyperparameters: HParams = HParams()

        self.net_configuration: dict = net_configuration
        model: SRL_final_MODEL = SRL_final_MODEL(hparams=hyperparameters, configurator=net_configuration).to(self.device)

        model.load_state_dict(torch.load('model/final_model_stored.pth', map_location=self.device))
        self.model: SRL_final_MODEL = model
        self.model.eval()  # set model in eval settings

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "pos_tags":
                            ["IN", "DT", "NN", ",", "NNP", "NNP", "CC", "DT", "NNS", "VBP", "IN", "DT", "JJ", "NNS", "IN", "VBG", "DT", "NN", "NN", "VBP", "RB", "VBN", "VBN", "."],
                        "dependency_heads":
                            ["10", "3", "1", "10", "6", "10", "6", "9", "7", "0", "10", "14", "14", "20", "14", "15", "19", "19", "16", "11", "20", "20", "22", "10"],
                        "dependency_relations":
                            ["ADV", "NMOD", "PMOD", "P", "TITLE", "SBJ", "COORD", "NMOD", "CONJ", "ROOT", "OBJ", "NMOD", "NMOD", "SBJ", "NMOD", "PMOD", "NMOD", "NMOD", "OBJ", "SUB", "TMP", "VC", "VC", "P"],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence. 
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence. 
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence. 
                    }
        """
        sent = {0: sentence}
        dataset_test: SRL_Dataset = SRL_Dataset(sentences=sent, labels=None, device=self.device, max_len=MAX_LEN, configurator=self.net_configuration, bert_model=self.bert_model, bert_tokenizer=self.bert_tokenizer)
        dataset_test.build_sample(self.vocab_words, self.vocab_pos_tags, self.vocab_lemmas, self.vocab_predicates, self.vocab_label, self.vocab_dependency_relations)
        batch_size: int = 32
        dataloader_test: DataLoader = DataLoader(dataset_test, batch_size=batch_size)
        return stud.srl_utility.print_output_docker(self.model, dataloader_test, self.vocab_label)[0]
