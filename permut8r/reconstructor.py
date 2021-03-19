class Reconstructor:
    """
    Turns detokens list into a string, tokenizes it again, and then re-creates the error label and undoes tokenization.
    This is needed because some operations will make the current error labels invalid, eg:
    1. Permutation
    DTK: ['大', 'き', 'く', 'なる'] ->  DEL chr 0 idx 3  ->  ['大', 'き', 'く', 'る']
    LAB:   1,    1,    1,    1              ->                1,    1,    1,    2
    2. Reconstruction
    Detokens to string: '大きくる'
    Re-tokenized during training: ['大', 'き', 'くる']
    Now, labels do not match - the labels should be 1, 1, 2, 0, but they were saved as 1, 1, 1, 2.
    Therefore, the Reconstructors job is to ensure that this does not happen, and that labels match tokenized strings
    when they are tokenized again during training.

    Params:
    ------
    tokenizer: huggingface transformers pre-trained tokenizer object:
        tokenizer object for converting strings into token lists, and token lists into de-tokenized string lists
    max_tok_length: int:
        the token length of the output sequences
    logging: bool:
        set to True to print logs for every step of the process
    """
    def __init__(self, tokenizer, max_tok_length, logging=True):
        """
        Creates an instance of the Reconstructor class.

        Params:
        ------
        tokenizer: huggingface transformers pre-trained tokenizer object:
            tokenizer object for converting strings into token lists, and token lists into de-tokenized string lists
        max_tok_length: int:
            the token length of the output sequences
        logging: bool:
            set to True to print logs for every step of the process
        """
        self.tokenizer = tokenizer
        self.logging = logging
        self.max_tok_length = max_tok_length

    def encode_to_ids(self, sentence):
        """ Converts a sentence into it's tokenized output. Returns the input_ids list only. """
        return self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_tok_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )['input_ids']

    @staticmethod
    def toks_to_line(toks):
        """ Recombines tokens into a string. """
        return ''.join(toks).replace('#', '')

    @staticmethod
    def expand_labels(detoks, labels):
        """ Expands a label to the length of each token. """
        result = ''
        for detok, label in zip(detoks, labels):
            result += str(label) * len(detok.replace('#', ''))
        return result

    @staticmethod
    def get_tok_lengths(detoks):
        """ Get the length of every detoken as save as a list. """
        return [(len(detok.replace('#', ''))) for detok in detoks]

    @staticmethod
    def get_tok_length_indices(detok_lengths):
        """
        Get the increasing index of every detoken length for use as a reference.
        Note that each value indicates a range(indices[i], indices[i + 1]) which applies to a detoken.
        """
        try:
            result = [detok_lengths[0]]
        except IndexError:
            print(detok_lengths)
            raise Exception
        for i, length in enumerate(detok_lengths[1:]):
            result.append(length + result[i])
        return result

    @staticmethod
    def repaint_labels(error_indices, retok_length_indices):
        """ Fill labels in the new, correct place for the valid portion of the label. """
        result = [1] * len(retok_length_indices)
        for error_index in error_indices:
            for i, tok_index in enumerate(retok_length_indices):
                if tok_index > error_index:
                    result[i] = 2
                    break
        return result

    @staticmethod
    def count_non_bert_tokens(total, tokens):
        # Counts the number of valid tokens which aren't CLS, PAD or SEP
        return total - (sum([x in [0, 2, 3] for x in tokens]))

    def reconstruct_line(self, detoks, err_label, num_valid):
        """ Given an input line and label, converts back to a sentence, then re-tokenizes and reallocates labels. """
        if self.logging:
            print(f'        ReX: Begin reconstruction ... ')

        line = self.toks_to_line(detoks[1:1 + num_valid])
        expanded_label = self.expand_labels(detoks[1:1 + num_valid], err_label[1:1 + num_valid])
        error_indices = [i for i, x in enumerate(expanded_label) if x == "2"]
        if self.logging:
            print(f'        ReX: LINE     : {line}')
            print(f'        ReX: EXPAND   : {expanded_label}')
            print(f'        ReX: ERRORIDX : {error_indices}')

        tokens = self.encode_to_ids(line)
        num_valid_retokens = self.count_non_bert_tokens(self.max_tok_length, tokens)
        retokens = self.tokenizer.convert_ids_to_tokens(tokens)

        if num_valid_retokens == 0:
            # Issue can arise if eg. all tokens get deleted in the sentence
            print(f'Num of valid retokens was 0 in {retokens}. Bailing to prevent error.')
            return False

        if self.logging:
            print(f'        ReX: deTOKENS : {detoks}')
            print(f'        ReX: reTOKENS : {retokens}')
            print(f'        ReX: VALIDS   : Previously: {num_valid}, now: {num_valid_retokens}')

        retok_lengths = self.get_tok_lengths(retokens[1:1 + num_valid_retokens])
        retok_length_indices = self.get_tok_length_indices(retok_lengths)
        if self.logging:
            print(f'        ReX: LENGTHS  : {retok_lengths}')
            print(f'        ReX: LENGTHIDX: {retok_length_indices}')

        new_valid_part_label = self.repaint_labels(error_indices, retok_length_indices)
        new_label = [0] + new_valid_part_label + (self.max_tok_length - 1 - num_valid_retokens) * [0]
        if self.logging:
            print(f'        ReX: ORIGLABEL: {err_label}')
            print(f'        ReX: NEWLABEL : {new_label}')

        return retokens, new_label, num_valid_retokens
