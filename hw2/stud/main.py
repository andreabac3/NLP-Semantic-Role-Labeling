# !/usr/bin/env python3


from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from transformers import BertTokenizer, BertModel, BertConfig

import stud.srl_utility
from stud.constant import *
import stud.trainer
from stud.srl_model import SRL_final_MODEL
from stud.srl_dataset import read_dataset, SRL_Dataset, build_vocab, build_vocab_roles

# I fix the seed to reproducibility reason
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE: str = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"

# Some path for the training phase
DATASET_PATH: str = '../../data/train.json'
DATASET_DEV_PATH: str = '../../data/dev.json'
DATASET_TEST_PATH: str = '../../data/test.json'
GLOVE_PATH: str = "../../model/glove.6B.300d.txt"  # pre-trained glove embeddings path

# read the dataset
sentences, labels = read_dataset(DATASET_PATH)
sentences_dev, labels_dev = read_dataset(DATASET_DEV_PATH)

# -- Initialize bert --
bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name, config=bert_config)

# -- net configuration -- It improve the code modularity
net_configuration: dict = net_configurator(use_bert_embeddings=USE_BERT_EMBEDDINGS, use_crf=USE_CRF, use_biaffine_layer=USE_BIAFFINE_LAYER, use_pretrained=USE_GLOVE, use_dependecy_heads=USE_DEPENDENCY_HEADS, use_predicates=False, use_syntagnet=USE_SYNTAGNET)

dataset_train: SRL_Dataset = SRL_Dataset(sentences, labels, device=DEVICE, configurator=net_configuration, bert_model=bert_model, bert_tokenizer=bert_tokenizer)

# creation of vocab starting from the train dataset
vocab_words: Vocab = build_vocab(dataset_train.sentences, 'words', min_freq=MIN_FREQUENCY)
vocab_pos_tags: Vocab = build_vocab(dataset_train.sentences, 'pos_tags')
vocab_lemmas: Vocab = build_vocab(dataset_train.sentences, 'lemmas', min_freq=MIN_FREQUENCY)
vocab_predicates: Vocab = build_vocab(dataset_train.sentences, 'predicates')
vocab_dependency_relations: Vocab = build_vocab(dataset_train.sentences, 'dependency_relations')
vocab_label: Vocab = build_vocab_roles(dataset_train.labels)

store_dataset_train: dict = {"vocab_words": vocab_words,
                             "vocab_pos_tags": vocab_pos_tags,
                             "vocab_lemmas": vocab_lemmas,
                             "vocab_predicates": vocab_predicates,
                             "vocab_label": vocab_label,
                             "vocab_dependency_relations": vocab_dependency_relations
                             }

if USE_STORED_DATASET:
    # if true we load the train dataset without computing it again
    dataset_train = torch.load("../../model/train_dataset_stored.pth")  # Load the train dataset
else:
    # calculate the train dataset and save it
    torch.save(store_dataset_train, "../../model/dict_vocabs.pth")  # Save a copy of all vocabs
    dataset_train.build_sample(vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label, vocab_dependency_relations)  # feature encoding phase
    torch.save(dataset_train, "../../model/train_dataset_stored.pth")

# -- Hyperparameters class --
@dataclass
class HParams:
    label_vocabulary: Vocab = vocab_label
    vocab_size_words: int = len(vocab_words)
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
    bert_hidden_dim: int = bert_config.hidden_size
    bert_lstm_num_layers: int = 2
    num_classes: int = len(vocab_label)
    bidirectional: bool = True
    num_layers: int = 2
    dropout: float = 0.3
    lstm_dropout: float = 0.3
    biaffine_lstm_num_layers: int = 2
    vocab_size_pos_tags: int = len(vocab_pos_tags)
    vocab_size_lemmas: int = len(vocab_lemmas)
    vocab_size_dependency_relations: int = len(vocab_dependency_relations)
    vocab_size_predicates: int = len(vocab_predicates)
    device: str = DEVICE


hyperparameters: HParams = HParams()

dataloader_train: DataLoader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

# Creation of the dev dataset
dataset_dev: SRL_Dataset = SRL_Dataset(sentences_dev, labels_dev, device=DEVICE, max_len=dataset_train.max_len, configurator=net_configuration, bert_model=bert_model, bert_tokenizer=bert_tokenizer)
dataset_dev.build_sample(vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label, vocab_dependency_relations)
dataloader_dev: DataLoader = DataLoader(dataset_dev, batch_size=BATCH_SIZE_VALID_TEST)

if USE_GLOVE:
    '''
    Load the GloVe embeddings
    '''
    from torchtext.vocab import Vectors

    vectors = Vectors(GLOVE_PATH, cache="./")
    pretrained_embeddings = torch.randn(len(vocab_words), vectors.dim)
    initialised = 0
    for i, w in enumerate(vocab_words.itos):
        if w in vectors.stoi:
            initialised += 1
            vec = vectors.get_vecs_by_tokens(w)
            pretrained_embeddings[i] = vec

    pretrained_embeddings[vocab_words[pad_token]] = torch.zeros(vectors.dim)
    hyperparameters.embedding_dim = vectors.dim
    hyperparameters.glove_embeddings = pretrained_embeddings
    hyperparameters.vocab_size_words = len(vocab_words)
    print("VECTOR DIM", vectors.dim)
    print("initialised embeddings {}".format(initialised))
    print("random initialised embeddings {} ".format(len(vocab_words) - initialised))

print(hyperparameters)
print(net_configuration)

model: SRL_final_MODEL = SRL_final_MODEL(hparams=hyperparameters, configurator=net_configuration).to(DEVICE)
trainer = stud.trainer.Trainer(
    model=model,
    loss_function=torch.nn.CrossEntropyLoss(ignore_index=vocab_label[pad_token]),
    optimizer=torch.optim.Adam(model.parameters()),
    label_vocab=vocab_label,
    device=DEVICE,
)

if USE_STORED:
    # if true, load the state dict of the model otherwise performe the train phase
    model.load_state_dict(torch.load('../../model/final_model_stored.pth', map_location=DEVICE))  # final
else:
    _, dict_loss = trainer.train(dataloader_train, dataloader_dev, epochs=EPOCHS, plot_loss=PLOT_LOSS)
    stud.srl_utility.plot_loss(dict_loss)  # plot the train and dev loss
    # store the model
    torch.save(model.state_dict(), "../../model/model_stored.pth")
    torch.save(trainer.optimizer.state_dict(), "../../model/model_stored_optimizer.pth")

if USE_TEST_EVALUATION:
    # If true, performe a test evaluation
    sentences_test, labels_test = read_dataset(DATASET_TEST_PATH)
    dataset_test: SRL_Dataset = SRL_Dataset(sentences_test, labels_test, device=DEVICE, max_len=dataset_train.max_len, configurator=net_configuration, bert_model=bert_model, bert_tokenizer=bert_tokenizer)
    dataset_test.build_sample(vocab_words, vocab_pos_tags, vocab_lemmas, vocab_predicates, vocab_label, vocab_dependency_relations)
    dataloader_test: DataLoader = DataLoader(dataset_test, batch_size=BATCH_SIZE_VALID_TEST)
    test_predictions: dict = stud.srl_utility.print_output_docker(model, dataloader_test, vocab_label)
    test_arg_identification = stud.srl_utility.evaluate_argument_identification(labels_test, test_predictions)
    print("TEST: IDENTIFICATION -> ", test_arg_identification)
    test_arg_classification = stud.srl_utility.evaluate_argument_classification(labels_test, test_predictions)
    print("TEST: CLASSIFICATION -> ", test_arg_classification)
    stud.srl_utility.calculate_confusion_matrix(labels_test, test_predictions, 'TEST')  # plot the normalized confusion matrix

# DEV evaluation
dev_predictions: dict = stud.srl_utility.print_output_docker(model, dataloader_dev, vocab_label)
dev_arg_identification = stud.srl_utility.evaluate_argument_identification(labels_dev, dev_predictions)
print("DEV: IDENTIFICATION -> ", dev_arg_identification)
dev_arg_classification = stud.srl_utility.evaluate_argument_classification(labels_dev, dev_predictions)
print("DEV: CLASSIFICATION -> ", dev_arg_classification)
stud.srl_utility.calculate_confusion_matrix(labels_dev, dev_predictions, 'DEV')  # plot the normalized confusion matrix
