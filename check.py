import sys
try:
    sys.argv[1]
except IndexError:
    print('Please provide an input sentence after "python check.py "')
    exit()

import numpy as np
from transformers import AutoModelForTokenClassification, BertJapaneseTokenizer

model = AutoModelForTokenClassification.from_pretrained('bunpo-check', num_labels=3)
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


def encode(sentence):
    """ Converts a sentence into it's tokenized output, ready for input to the model. """
    return tokenizer(
        sentence,
        add_special_tokens=True,
        max_length=48,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )


def check_string(string):
    """ Passes tokenized string through the model, returns detokens and predictions. """
    encoding = encode(string)
    output = model(
        encoding['input_ids'],
        encoding['attention_mask']
    ).logits.cpu().detach().numpy()
    detokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'].numpy()[0])
    predictions = np.argmax(output, axis=2)[0]
    return detokens, predictions


def check_and_print(string):
    """ Check a string and print the output for each token. """
    detoks, preds = check_string(string)
    print(f'Checking string: {string}')
    print('Label key: 0: unimportant token, 1: correct, 2: error detected.')
    print('Token    | Label')
    for d, p in zip(detoks, preds):
        print(f'{d:<8}| {p}')


sentence = sys.argv[1]
check_and_print(sentence)
