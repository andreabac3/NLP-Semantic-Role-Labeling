# !/usr/bin/env python3

'''
This file is used to store all constant in the project.
'''

USE_STORED: bool = True  # if true the train will not performed, we use the pre-trained model
USE_STORED_DATASET: bool = True  # if true will be used the pre calculated dataset
USE_GPU: bool = True  # if false, turn off the GPU
USE_TEST_EVALUATION: bool = True
SEED: int = 42  # fix the seed for reproducibility reason

USE_GLOVE: bool = True  # if true, the glove embeddings will loaded.
USE_BERT_EMBEDDINGS: bool = True  # if true, the bert embeddings will be calculated
USE_DEPENDENCY_HEADS: bool = True  # if true, the GCN will be used
USE_BIAFFINE_LAYER: bool = True  # if true, the attention layer will be used

USE_CRF: bool = False  # not used
USE_SYNTAGNET: bool = False  # not used

MIN_FREQUENCY: int = 2

EPOCHS: int = 2  # 11

BATCH_SIZE: int = 128  # 64 # 128
BATCH_SIZE_TEST: int = 32
BATCH_SIZE_VALID_TEST: int = 32

MAX_LEN: int = 143  # an integer to indicate the pre calculated max length.

PLOT_LOSS: bool = True  # if true, at the end of training you can see loss vs epochs and f1 vs epochs plot

unk_token: str = '<unk>'
pad_token: str = '<pad>'

model_name: str = 'bert-base-cased'



def net_configurator(use_pretrained: bool, use_bert_embeddings: bool, use_dependecy_heads: bool, use_biaffine_layer: bool, use_crf: bool, use_predicates: bool, use_syntagnet: bool) -> dict:
    '''
    Return a config used in the dataset class and in the model class, in order to enable/disable some layer, improve code modularity
    '''
    config: dict = {"use_syntagnet": use_syntagnet, "use_predicates": use_predicates, "use_pretrained": use_pretrained, "use_bert_embeddings": use_bert_embeddings, "use_dependecy_heads": use_dependecy_heads, "use_biaffine_layer": use_biaffine_layer, "use_crf": use_crf}
    return config
