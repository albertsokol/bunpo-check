class Swapper:
    """
    The swapper swaps 2 tokens with each other. A choice whether to swap left or right is made randomly, except for
    tokens at the start or end of the sentence, which can only be swapped with the token next to them.

    Params:
    ------
    rng: numpy default_rng() object:
        random number generator that is shared across all the permutators
    logging: bool:
        set to True to print logs for every step of the process
    """
    def __init__(self, rng, logging=True):
        """
        Creates an instance of the Swapper class.

        Params:
        ------
        rng: numpy default_rng() object:
            random number generator that is shared across all the permutators
        logging: bool:
        set to True to print logs for every step of the process
        """
        self.rng = rng
        self.logging = logging

    def swap(self, detokens, original_detokens, err_label, index, num_valid_tokens):
        """
        Swaps tokens and returns the permuted list of tokens.

        Params:
        ------
        detokens: list:
            the current token-split list of strings, obtained from converting the tokenized sentence back into characters
        original_detokens: list:
            the original detokens list, obtained before applying any permutations
        err_label: list:
            the current list of 0, 1 and 2 labels assigned to each token
        index: int:
            the index of the token to be permuted
        num_valid_tokens: int:
            the number of non-padding, non-EOS, non-CLS etc. tokens in the string

        Returns:
        -------
        detokens: list:
           the updated list of the detokens, with some detokens swapped around
        err_label: list:
            the updated list of 0, 1 and 2 labels assigned to each token
        """
        if self.logging:
            print(f'SWAP  : Swapping index {index} of {num_valid_tokens}')

        # At start of sentence
        if index == 1:
            detokens[1], detokens[2] = detokens[2], detokens[1]
            err_label[1] = 2
            err_label[2] = 2
            if self.logging:
                print(f'sSTART: {detokens}')

        # At end of sentence
        elif index == num_valid_tokens:
            detokens[num_valid_tokens], detokens[num_valid_tokens - 1] = detokens[num_valid_tokens - 1], detokens[num_valid_tokens]
            err_label[num_valid_tokens] = 2
            err_label[num_valid_tokens - 1] = 2
            if self.logging:
                print(f'sEND  : {detokens}')

        # Somewhere in the middle
        else:
            if self.rng.uniform() > 0.5:
                detokens[index], detokens[index - 1] = detokens[index - 1], detokens[index]
                err_label[index] = 2
                err_label[index - 1] = 2
                if self.logging:
                    print(f'sLEFT : {detokens}')
            else:
                detokens[index], detokens[index + 1] = detokens[index + 1], detokens[index]
                err_label[index] = 2
                err_label[index + 1] = 2
                if self.logging:
                    print(f'sRIGHT: {detokens}')

        # Remove BOGUS error labels where the original and new detoken lists have ended up being the same as the original tokens
        for (i, new, original) in zip(range(1, 1 + num_valid_tokens), detokens[1:1 + num_valid_tokens], original_detokens[1:1 + num_valid_tokens]):
            if new == original:
                err_label[i] = 1

        return detokens, err_label
