# 文法ーCHECK: models and code
Hi! Here you can find the trained models for [bunpo-check.com](https://bunpo-check.com), as well as libraries for generating the data and carrying out the model training.

### Contents
 - [Permutations library: permut8r](#permut8r)
 - [Trained models available for download](#trained-models) 
 
### Still to be added 
 - [Model training code](#model-training)
 - Website JS/HTML/CSS 
 - Functions for interpreting output 
 
### Requirements
Will add soon! 

## permut8r 
This is a permutation library I wrote for creating random permutations of correct strings, for use as pre-training data.

**What does it do?**

```python
# Original string from Wikipedia
"その使用は1世紀に遡ることができ、5世紀中葉から現代に至るまでの変遷がわかる。"

# Output from permut8r
"その使用は1世紀に遡ることができ、5世紀中葉から現代に至るまでの変遷がわかる。,011111111111111111111111110000000000000000000000"
"その使用1世紀に遡ることができ、5世紀中た葉から現代に至るまでの変遷業がわかる。,012211111111112211111112211000000000000000000000"
``` 

A string is converted to tokens via the Huggingface tokenizer. Then, random permutations are applied to it from the following:
 - Deleting tokens or characters 
 - Swapping tokens around 
 - Swapping kanji for other kanji with the same reading
 - Swapping particles for other random particles
 - Inserting random kanji or particles

Each token is assigned a label of 0 (unimportant eg. PAD tokens), 1 (correct grammar) or 2 (error). The correct string is saved with a label sequence of only 0/1. The permuted string contains some new errors, and these are encoded with a label of 2 for the affected tokens. 

This allowed me to build a ~263M dataset of labelled sentences which reached a decent F0.5 score of 45.0 on a test set, which went a looong way to getting a good performance here.

## Model training
Will upload soon! 


## Trained models 
To use the trained model, you will need to use the Huggingface transformers library.

Step 1: download the model from [my Google drive](https://drive.google.com/drive/folders/1ur4GZV_E_Is4ehhNwLv4w2p02IC_ZnWa?usp=sharing) and save it to your working directory. 

Step 2: load the model. The tokenizer will be downloaded automatically from the Huggingface repository. 
```python
import numpy as np 
from transformers import AutoModelForTokenClassification, BertJapaneseTokenizer

model = AutoModelForTokenClassification.from_pretrained('bunpo-check', num_labels=3)
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
```

Step 3: define functions for checking your string using the model.
```python
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
```

Step 4: use the check and print function on your own input.
```python
check_and_print('⽂法ーCHECKは⼈⼯知能により⽂法が正しいか確かめられるサイトです。')
```
