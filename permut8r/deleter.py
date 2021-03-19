class Deleter:
    """
    Deletes either one character from a token, or the entire token (entire token has an equal chance of being deleted
    with respect to each individual character).

    Params:
    ------
    rng: numpy default_rng() object:
        random number generator that is shared across all the permutators
    logging: bool:
        set to True to print logs for every step of the process
    """
    def __init__(self, rng, logging=True):
        """
        Creates an instance of the Deleter class.

        Params:
        ------
        rng: numpy default_rng() object:
            random number generator that is shared across all the permutators
        logging: bool:
            set to True to print logs for every step of the process
        """
        self.rng = rng
        self.logging = logging

    def delete_roll(self, tok_count):
        """ Delete it all? Delete just 1? What will it be? Each character and 'ALL' have an equal chance. """
        ticket = self.rng.uniform()
        if ticket < 1. / (tok_count + 1):
            return 'ALL'
        else:
            return 'SINGLE'

    def delete(self, detokens, err_label, index, num_valid_tokens):
        """
        Delete from the detoken at the specified index.

        Params:
        ------
        detokens: list:
            the current token-split list of strings, obtained from converting the tokenized sentence back into characters
        err_label: list:
            the current list of 0, 1 and 2 labels assigned to each token
        index: int:
            the index of the token to be permuted
        num_valid_tokens: int:
            the number of non-padding, non-EOS, non-CLS etc. tokens in the string

        Returns:
        -------
        detokens: list:
           the updated list of the detokens, with a character or token deleted
        err_label: list:
            the updated list of 0, 1 and 2 labels assigned to each token
        """
        isolated_detoken = detokens[index].replace('#', '')

        if len(isolated_detoken) == 0:
            print(f'Detoken {isolated_detoken} at {index} had length 0 and so returning. Full detokens: {detokens}')
            return detokens, err_label

        if isolated_detoken in '。！？!?.':
            if self.logging:
                print(f'DELETE: Skipped deleting an EOS character.')
            return detokens, err_label

        # If it's a len 1 token then just remove the text
        if len(isolated_detoken) == 1:
            detokens[index] = ''
            # If it's at the start then just update the next label
            if index == 1:
                err_label[index + 1] = 2
            # If it's at the end then update the second to last label
            elif index == num_valid_tokens:
                err_label[index - 1] = 2
            # Otherwise update the label at either side of it
            else:
                err_label[index + 1] = 2
                err_label[index - 1] = 2
            if self.logging:
                print(f'DELETE: Removed token {isolated_detoken} at index {index}. (LEN 1 DELETE).')
            return detokens, err_label

        # Else roll to decide whether to delete all, or just a single character
        else:
            ticket = self.delete_roll(len(isolated_detoken))

            if ticket == 'ALL':
                detokens[index] = ''
                # If it's at the start then just update the next label
                if index == 1:
                    err_label[index + 1] = 2
                # If it's at the end then update the second to last label
                elif index == num_valid_tokens:
                    err_label[index - 1] = 2
                # Otherwise update the label at either side of it
                else:
                    err_label[index + 1] = 2
                    err_label[index - 1] = 2
                if self.logging:
                    print(f'DELETE: Removed token {isolated_detoken} at index {index}. (ALL DELETE).')
                return detokens, err_label

            else:
                # Pick the index of the character that will be deleted.
                intra_token_index = self.rng.integers(0, len(isolated_detoken))
                new_detoken = isolated_detoken[:intra_token_index] + isolated_detoken[1 + intra_token_index:]

                if self.logging:
                    print(f'DELETE: Removed {isolated_detoken[intra_token_index]} from {isolated_detoken} at intra-token index {intra_token_index} forming new de-token {new_detoken} for index {index}. (1 in MULTI DELETE).')

                detokens[index] = new_detoken
                err_label[index] = 2
                return detokens, err_label
