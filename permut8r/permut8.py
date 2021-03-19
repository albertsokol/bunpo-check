import json
from time import time

import MeCab
import numpy as np
from transformers import BertJapaneseTokenizer

from deleter import Deleter
from inserter import Inserter
from kanjiking import KanjiKing
from reconstructor import Reconstructor
from swapper import Swapper


def encode(sentence):
    """ Converts a sentence into it's tokenized output. Returns a dictionary with input_ids and attention_mask. """
    return tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_tok_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True
    )


def lotto():
    """ Eyyy step right up, step right up, get your lucky tickets here. Contains weightings for permutations. """
    ticket_no = rng.uniform()

    if ticket_no <= 0.15:
        return 'SWAP'
    elif ticket_no <= 0.45:
        return 'DELETE'
    elif ticket_no <= 0.75:
        return 'INSERT'
    else:
        return 'KANJI'


def count_non_bert_tokens(total, tokens):
    # Counts the number of valid tokens which aren't CLS, PAD or SEP
    return total - (sum([x in [0, 2, 3] for x in tokens]))


if __name__ == '__main__':
    # ============ HYPERPARAMETERS ============
    # The probability that a token will be left unaltered
    no_perm_probability = 6. / 7
    # The token length of the output sequences
    max_tok_length = 48
    # =========================================

    with open('kanji-dictionary.json', 'r') as kj:
        # A dictionary with katakana readings as keys, and kanji with that reading as a list of values (常用漢字)
        kanji_dictionary = json.loads(kj.read())
    with open('frequency-list.json', 'r') as fl:
        frequency_dict = json.loads(fl.read())

    # Set up tagger, tokenizer, rng
    tagger = MeCab.Tagger('-r /dev/null -d /home/y4tsu/anaconda3/lib/python3.8/site-packages/unidic_lite/dicdir -Odump')
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    rng = np.random.default_rng()
    start = time()

    # Create permutator objects
    logging = False
    reconstructor = Reconstructor(tokenizer, max_tok_length, logging=False)
    swapper = Swapper(rng, logging=logging)
    kk = KanjiKing(rng, tagger, kanji_dictionary, frequency_dict, logging=logging)
    inserter = Inserter(rng, kanji_dictionary, frequency_dict, logging=logging)
    deleter = Deleter(rng, logging=logging)

    # Progress information - could do with some improvements
    checkpoint = 1_570_000  # 1% of the number of lines in the file
    count_read = 0

    print('YO yo YO let\'s gooooooooo')

    with open('./testing.txt', 'r') as r:
        with open('permutations.txt', 'w') as w:
            w.write('')
        with open('permutations.txt', 'a') as a:
            while True:
                bail = False
                count_read += 1

                try:
                    line = r.readline()
                except UnicodeDecodeError:
                    if logging:
                        print('Invalid byte read -- skipped')
                    continue

                if count_read % checkpoint == 0:
                    norm_time = (time() - start) / (count_read // checkpoint)
                    print(f'\rRead {(count_read // checkpoint)}% of file, estimate '
                          f'{100 * norm_time - (count_read // checkpoint) * norm_time:.1f} seconds remaining.',
                          end="", flush=True)

                if not line:
                    print(f'End of file reached! Read {count_read} lines. Took {time() - start:.1f} seconds.')
                    print(f'{"Lines read:":<18}{count_read:>12,}')
                    break

                # PRE-CLEAN
                line = line.strip().strip('\n')

                # TOKENIZE, DE-TOKENIZE, MAKE LABELS
                tokens = encode(line)['input_ids']
                detokens = tokenizer.convert_ids_to_tokens(tokens)
                original_detokens = detokens.copy()
                num_valid_tokens = count_non_bert_tokens(max_tok_length, tokens)
                corr_label = [0] + [1] * num_valid_tokens + (max_tok_length - 1 - num_valid_tokens) * [0]
                err_label = corr_label.copy()

                # ROLL
                num_to_permutate = sum(np.where(rng.uniform(size=num_valid_tokens) > no_perm_probability, 1, 0))

                if logging:
                    print(f'New line with {num_valid_tokens} valid tokens selected.')
                    print(f'Rolled {num_to_permutate} permutations.')

                if num_to_permutate == 0:
                    if logging:
                        print('No permutation was drawn. :(')
                        print('-----')
                    continue

                if logging:
                    print(f'ORIGNL: {detokens}')

                # PERMUTATE
                for _ in range(num_to_permutate):
                    # Pick the index that will be permuted
                    curr_index_to_permutate = rng.integers(1, 1 + num_valid_tokens)

                    # Get your lucky ticket
                    ticket = lotto()
                    if logging:
                        print(f'Ticket: Lucky ticket {ticket} rolled for {detokens[curr_index_to_permutate]} at index {curr_index_to_permutate}')

                    # Spend your ticket and update the detoks and err_label with the result
                    if ticket == 'SWAP':
                        detokens, err_label = swapper.swap(detokens, original_detokens, err_label, curr_index_to_permutate, num_valid_tokens)
                    elif ticket == 'KANJI':
                        detokens, err_label = kk.kanji(detokens, err_label, curr_index_to_permutate)
                    elif ticket == 'INSERT':
                        detokens, err_label = inserter.insert(detokens, err_label, curr_index_to_permutate)
                    else:
                        detokens, err_label = deleter.delete(detokens, err_label, curr_index_to_permutate, num_valid_tokens)

                    # If the detokens have all been deleted then exit here to prevent errors (unlikely but possible)
                    if (detokens == ['CLS'] + [''] + ['SEP'] + (['PAD'] * (max_tok_length - 3))) \
                            or (detokens == ['CLS'] + (['SEP'] * 2) + (['PAD'] * (max_tok_length - 3))):
                        bail = True
                        if logging:
                            print(f'Detokens {detokens} was empty and so line {line} was bailed.')
                        break

                    # Reconstruct the line, then split again into de-tokens and update the label accordingly
                    result = reconstructor.reconstruct_line(detokens, err_label, num_valid_tokens)
                    if not result:
                        bail = True
                        if logging:
                            print(f'Bailing at error ... ')
                        break
                    else:
                        detokens, err_label, num_valid_tokens = result
                        if logging:
                            print(f'DETOKS: {detokens}')
                            print(f'ERR_LB: {err_label}')

                if bail:
                    continue

                # If the final detokens are the same as the original, it's a Bogus Transform™
                if detokens == original_detokens:
                    if logging:
                        print(f'BOGUS TRANSFORM! Original and permuted detokens are the same.')
                else:
                    # Otherwise all is good so write 2 lines to the output file
                    corr_label = "".join([str(x) for x in corr_label])
                    new_line = reconstructor.toks_to_line(detokens[1:1 + num_valid_tokens])
                    new_label = "".join([str(x) for x in err_label])
                    line = line.replace(',', '、')  # Replace commas with JP commas to prevent CSV read errors
                    new_line = new_line.replace(',', '、')
                    if logging:
                        print(f'OUTPUT: Original line: {line}')
                        print(f'OUTPUT: New line     : {new_line}')
                        print(f'OUTPUT: Correct label: {corr_label}')
                        print(f'OUTPUT: Error label  : {new_label}')
                    a.write(f'{line},{corr_label}\n')  # Save the original line with a label of all 0s and 1s
                    a.write(f'{new_line},{new_label}\n')  # Save the permuted line with labels of 0s, 1s and 2s

                if logging:
                    print('-----')
