class KanjiKing:
    """
    The Kanji King has two important jobs:
    1. Swapping a Kanji for a frequency-weighted alternative Kanji of the same reading eg. 高 -> 後
    2. Swapping a particle for another random particle eg. は -> が
    The type of transformation depends on the contents of the randomly selected token. If the token contains a single
    kanji only, eg. 高, then this kanji will be swapped. If the token contains multiple kanji eg. 高速, one of the
    kanji will be randomly selected, then swapped. If the token contains only a particle eg. は, then that particle
    will be swapped.

    Params:
    ------
    rng: numpy default_rng() object:
        random number generator that is shared across all the permutators
    tagger: MeCab tagger object:
        the MeCab parser with correct dictionary selected and output format set to dump, shared across permutators
    kanji_dictionary: dict:
        a dictionary with katakana readings as keys, and kanji with that reading as a list of values (常用漢字)
    frequency_dict: dict:
        a dictionary with kanji as keys, and their frequency in JP Wikipedia as values
    logging: bool:
        set to True to print logs for every step of the process
    """
    def __init__(self, rng, tagger, kanji_dictionary, frequency_dict, logging=True):
        """
        Create an instance of the KanjiKing class.

        Params:
        ------
        rng: numpy default_rng() object:
            random number generator that is shared across all the permutators
        tagger: MeCab tagger object:
            the MeCab parser with correct dictionary selected and output format set to dump, shared across permutators
        kanji_dictionary: dict:
            a dictionary with katakana readings as keys, and kanji with that reading as a list of values (常用漢字 only)
        frequency_dict: dict:
            a dictionary with kanji as keys, and their frequency in JP Wikipedia as values
        logging: bool:
            set to True to print logs for every step of the process
        """
        self.kanji_dictionary = kanji_dictionary
        self.frequency_dict = frequency_dict

        self.rng = rng
        self.tagger = tagger
        self.katakana = "ァアアィイイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペ" \
                        "ホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヽヾー"
        self.particles_short = "はがのにとでをもへやかば"
        self.logging = logging

    def kanji_lotto(self, homonyms):
        """ Takes a single kanji, gets the reading, assigns a frequency-weighted misspelling with the same reading. """
        tombola = [self.frequency_dict[homonyms[0]]]
        for i, homonym in enumerate(homonyms[1:]):
            frequency = self.frequency_dict[homonym]
            tombola.append(frequency + tombola[i])

        kanji_ticket_no = self.rng.integers(0, tombola[-1])

        for i, cumul_frequency in enumerate(tombola):
            if kanji_ticket_no < cumul_frequency:
                return homonyms[i]

    def get_homonyms(self, reading):
        """ Returns all the homonyms for a single reading as a list. Returns False if there are none, or only one. """
        try:
            homonyms = self.kanji_dictionary[reading]
            if self.logging:
                print(f'HMNYMS: {homonyms}')
            return homonyms if len(homonyms) > 1 else False
        except KeyError:
            return False

    def get_new_kanji_token(self, kanji, reading):
        # Get the homonyms for the token: return if there are no others
        homonyms = self.get_homonyms(reading)
        if not homonyms:
            if self.logging:
                print(f'KANJI : No other homonyms were found.')
            return False

        # Draw a new kanji with the same reading: return if the same kanji is picked
        new_kanji = self.kanji_lotto(homonyms)
        if new_kanji == kanji:
            if self.logging:
                print(f'KANJI : Tombola drew same kanji!')
            return False

        # Otherwise return the new kanji
        return new_kanji

    def particle_lotto(self):
        """ Choose an unweighted random particle from the particle list. """
        return self.particles_short[self.rng.integers(0, len(self.particles_short))]

    def kanji(self, detokens, err_label, index):
        """
        Swaps a kanji for a kanji with another reading, or particle for a particle with another reading.

        Params:
        ------
        detokens: list:
            the current token-split list of strings, obtained from converting the tokenized sentence back into characters
        err_label: list:
            the current list of 0, 1 and 2 labels assigned to each token
        index: int:
            the index of the token to be permuted

        Returns:
        -------
        detokens: list:
           the updated list of the detokens, with a kanji or particle swap applied
        err_label: list:
            the updated list of 0, 1 and 2 labels assigned to each token
        """
        isolated_detoken = detokens[index].replace('#', '')

        if len(isolated_detoken) == 0:
            print(f'Detoken {isolated_detoken} at {index} had length 0 and so returning. Full detokens: {detokens}')
            return detokens, err_label

        # Don't mess wiv da katakana
        if all([x in self.katakana for x in isolated_detoken]):
            if self.logging:
                print(f'KANJI : Fully katakana token was ignored.')
            return detokens, err_label

        # If it's a single particle then do a particle swapsie
        if isolated_detoken in self.particles_short and len(isolated_detoken) == 1:
            new_particle = self.particle_lotto()
            if new_particle == isolated_detoken:
                if self.logging:
                    print(f'KANJI : Particle swapsie failed! Not gonna swap {isolated_detoken} for {new_particle}, buster!')
                return detokens, err_label

            detokens[index] = new_particle
            err_label[index] = 2
            if self.logging:
                print(f'KANJI : Particle swapsie of {isolated_detoken} for {new_particle} at {index}. (PARTICLE EXCHANGE)')
            return detokens, err_label

        # Parse the token with MeCab
        try:
            tag = self.tagger.parse(isolated_detoken).split('\n')[1].split(',')
            # Get the type and reading
            tok_type = tag[12]
            tok_reading = tag[6]
        except IndexError:
            if self.logging:
                print(f'KANJI : Invalid token selected, skipping kanji permute. (PRE)')
            return detokens, err_label

        if self.logging:
            print(f'KANJI : Detoken {isolated_detoken} of type {tok_type} with reading {tok_reading} was selected.')

        # Deal with kanji if selected
        if tok_type == '漢' or tok_type == '固':
            # Deal with single kanji in a token
            if len(isolated_detoken) == 1:
                # Return if unable to get a new kanji
                new_kanji = self.get_new_kanji_token(isolated_detoken, tok_reading)
                if not new_kanji:
                    return detokens, err_label

                # Otherwise update the corresponding index and set the label to 2 then return
                if self.logging:
                    print(f'KANJI : Replaced {isolated_detoken} at index {index} with {new_kanji}. (1:1 EXCHANGE).')
                detokens[index] = new_kanji
                err_label[index] = 2

                return detokens, err_label

            # Deal with multiple kanji in a token
            else:
                # Select a random kanji inside the token
                intra_token_index = self.rng.integers(0, len(isolated_detoken))
                selected_kanji = isolated_detoken[intra_token_index]

                # Parse the new kanji with MeCab
                try:
                    tag = self.tagger.parse(selected_kanji).split('\n')[1].split(',')
                    tok_reading = tag[6]
                except IndexError:
                    if self.logging:
                        print(f'KANJI : Invalid token selected, skipping kanji permute. (MULTI).')
                    return detokens, err_label

                if self.logging:
                    print(f'KANJI : Shuffle selected {selected_kanji} with reading {tok_reading} at intra-token index {intra_token_index}')

                # Return if unable to get a new kanji
                new_kanji = self.get_new_kanji_token(selected_kanji, tok_reading)
                if not new_kanji:
                    return detokens, err_label

                # Otherwise update the corresponding index and set the label to 2 then return
                if self.logging:
                    print(f'KANJI : Replaced {selected_kanji} at index {index} with {new_kanji}. (1 in MULTI EXCHANGE).')
                new_detoken = isolated_detoken[:intra_token_index] + new_kanji + isolated_detoken[intra_token_index + 1:]
                detokens[index] = new_detoken
                err_label[index] = 2

                return detokens, err_label

        if self.logging:
            print(f'KANJI : No permutations were made.')
        return detokens, err_label
