# Student: you can use this file to prepare your environment once (and avoid re-downloading every time you start test.sh)
# e.g. if you use transformers, nltk, spacy, etc you can download everything needed by adding the proper code here!
# all the code in here will be executed only once (unless you change this file or requirements.txt)
# NOTE: DO NOT put anything referring to your own files here (no imports from stud), it won't work!
# NOTE: DO NOT declare anything here that you want to use in your stud folder!

from transformers import BertTokenizer, BertModel, BertConfig

model_name: str = 'bert-base-cased'
tokenizer_name: str = 'bert-base-cased'

bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name, config=bert_config)
