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
See requirements.txt 

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

Step 1: download the model from [my Google drive](https://drive.google.com/drive/folders/1ur4GZV_E_Is4ehhNwLv4w2p02IC_ZnWa?usp=sharing) and save the folder to your working directory. 

Step 2: run check.py to check your sentence.

```bash
python check.py "⽂法ーCHECKは⼈⼯知能により⽂法が正しいか確かめられるサイトです。"
```
