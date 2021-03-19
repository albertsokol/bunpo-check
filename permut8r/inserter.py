class Inserter:
    """
    Inserts a random kanji or particle to some part of a token.

    Params:
    ------
    rng: numpy default_rng() object:
        random number generator that is shared across all the permutators
    kanji_dictionary: dict:
        a dictionary with katakana readings as keys, and kanji with that reading as a list of values (常用漢字)
    frequency_dict: dict:
        a dictionary with kanji as keys, and their frequency in JP Wikipedia as values
    logging: bool:
        set to True to print logs for every step of the process
    """
    def __init__(self, rng, kanji_dictionary, frequency_dict, logging=True):
        """
        Creates an instance of the Inserter class.

        Params:
        ------
        rng: numpy default_rng() object:
            random number generator that is shared across all the permutators
        kanji_dictionary: dict:
            a dictionary with katakana readings as keys, and kanji with that reading as a list of values (常用漢字)
        frequency_dict: dict:
            a dictionary with kanji as keys, and their frequency in JP Wikipedia as values
        logging: bool:
            set to True to print logs for every step of the process
        """
        self.rng = rng
        self.kanji_dictionary = kanji_dictionary
        self.frequency_dict = frequency_dict

        self.kanji_probability = 0.1  # Probability of randomly selecting a Kanji to insert, instead of a particle

        # List of possible particles to add
        self.particles = 'をにではがおいてたるうくとなもへやばのか'
        # Precalculated list of particle frequencies in JP Wikipedia
        self.particle_frequencies = [63395, 92988, 64211, 77973, 63217, 8690, 66374, 62289, 69530, 99548, 18830,
                                     15895, 69467, 55615, 32417, 2741, 10913, 6204, 140392, 26798]
        # Precalculated cumulative version of the above list
        self.cumul_particle_frequencies = [63395, 156383, 220594, 298567, 361784, 370474, 436848, 499137, 568667,
                                           668215, 687045, 702940, 772407, 828022, 860439, 863180, 874093, 880297,
                                           1020689, 1047487]
        self.joyo = list(self.frequency_dict.keys())  # List of 常用漢字

        # Calculate cumulative kanji frequencies
        self.cumul_kanji_frequencies = [frequency_dict[self.joyo[0]]]
        for i, key in enumerate(self.joyo[1:]):
            frequency = self.frequency_dict[key]
            self.cumul_kanji_frequencies.append(self.cumul_kanji_frequencies[i] + frequency)

        self.logging = logging

    def insert_lotto(self):
        """ Lucky lotto!! What's it gonna be this time ... """
        ticket = self.rng.uniform()
        if ticket < self.kanji_probability:
            return 'KANJI'
        else:
            return 'PARTICLE'

    def get_direction_ticket(self):
        """ Roll whether to insert to the left or the right of the current character. """
        return 'LEFT' if self.rng.uniform() < 0.5 else 'RIGHT'

    def get_random_particle(self):
        """ Pick a random particle from the list, which will be inserted. """
        particle_ticket_no = self.rng.integers(0, self.cumul_particle_frequencies[-1])

        for i, cumul_frequency in enumerate(self.cumul_particle_frequencies):
            if particle_ticket_no < cumul_frequency:
                if self.logging:
                    print(f'INSERT: Tombola drew {self.particles[i]} for ticket no {particle_ticket_no} of cumul_frequencies {self.cumul_particle_frequencies}')
                return self.particles[i]

    def get_random_kanji(self):
        """ Pick a random kanji from the frequency list weighted by frequencies. """
        kanji_ticket_no = self.rng.integers(0, self.cumul_kanji_frequencies[-1])

        for i, cumul_frequency in enumerate(self.cumul_kanji_frequencies):
            if kanji_ticket_no < cumul_frequency:
                if self.logging:
                    print(f'INSERT: Tombola drew {self.joyo[i]} for ticket no {kanji_ticket_no} of cumul_frequencies {self.cumul_kanji_frequencies}')
                return self.joyo[i]

    def insert(self, detokens, err_label, index):
        """
        Insert a weighted random kanji or particle in the given token.

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
           the updated list of the detokens, with new kanji/particles added to the token at the given index
        err_label: list:
            the updated list of 0, 1 and 2 labels assigned to each token
        """
        ticket = self.insert_lotto()

        isolated_detoken = detokens[index].replace('#', '')
        # Check that the token does not consist only of #
        if len(isolated_detoken) == 0:
            if self.logging:
                print(f'Detoken {isolated_detoken} at {index} had length 0 and so returning. Full detokens: {detokens}')
            return detokens, err_label

        # If it's a particle that was drawn
        if ticket == 'PARTICLE':
            new_particle = self.get_random_particle()
            # Deal with length 1 token
            if len(isolated_detoken) == 1:
                direction_ticket = self.get_direction_ticket()
                if direction_ticket == 'LEFT':
                    if self.logging:
                        print(f'INSERT: Inserting {new_particle} to the LEFT of {isolated_detoken} at index {index}. (1:1 PARTICLE)')
                    new_detoken = new_particle + isolated_detoken
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label
                else:
                    if self.logging:
                        print(f'INSERT: Inserting {new_particle} to the RIGHT of {isolated_detoken} at index {index}. (1:1 PARTICLE)')
                    new_detoken = isolated_detoken + new_particle
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label

            # Deal with multiple character token
            else:
                # Pick a random index to add before or after
                intra_token_index = self.rng.integers(0, len(isolated_detoken))
                direction_ticket = self.get_direction_ticket()
                if direction_ticket == 'LEFT':
                    if self.logging:
                        print(f'INSERT: Inserting {new_particle} to the LEFT of {isolated_detoken[intra_token_index]} at intra-token index {intra_token_index} of {isolated_detoken}. (1 in MULTI PARTICLE)')
                    new_detoken = isolated_detoken[:intra_token_index] + new_particle + isolated_detoken[intra_token_index:]
                    if self.logging:
                        print(f'INSERT: New detoken: {new_detoken}')
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label
                else:
                    if self.logging:
                        print(f'INSERT: Inserting {new_particle} to the RIGHT of {isolated_detoken[intra_token_index]} at intra-token index {intra_token_index} of {isolated_detoken}. (1 in MULTI PARTICLE)')
                    new_detoken = isolated_detoken[:intra_token_index + 1] + new_particle + isolated_detoken[1 + intra_token_index:]
                    if self.logging:
                        print(f'INSERT: New detoken: {new_detoken}')
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label

        # Else it's a kanji that was drawn
        else:
            new_kanji = self.get_random_kanji()
            # Deal with length 1 token
            if len(isolated_detoken) == 1:
                direction_ticket = self.get_direction_ticket()
                if direction_ticket == 'LEFT':
                    if self.logging:
                        print(f'INSERT: Inserting {new_kanji} to the LEFT of {isolated_detoken} at index {index}. (1:1 KANJI)')
                    new_detoken = new_kanji + isolated_detoken
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label
                else:
                    if self.logging:
                        print(f'INSERT: Inserting {new_kanji} to the RIGHT of {isolated_detoken} at index {index}. (1:1 KANJI)')
                    new_detoken = isolated_detoken + new_kanji
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label

            # Deal with multiple character token
            else:
                # Pick a random index to add before or after
                intra_token_index = self.rng.integers(0, len(isolated_detoken))
                direction_ticket = self.get_direction_ticket()
                if direction_ticket == 'LEFT':
                    if self.logging:
                        print(f'INSERT: Inserting {new_kanji} to the LEFT of {isolated_detoken[intra_token_index]} at intra-token index {intra_token_index} of {isolated_detoken}. (1 in MULTI KANJI)')
                    new_detoken = isolated_detoken[:intra_token_index] + new_kanji + isolated_detoken[intra_token_index:]
                    if self.logging:
                        print(f'INSERT: New detoken: {new_detoken}')
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label
                else:
                    if self.logging:
                        print(f'INSERT: Inserting {new_kanji} to the RIGHT of {isolated_detoken[intra_token_index]} at intra-token index {intra_token_index} of {isolated_detoken}. (1 in MULTI KANJI)')
                    new_detoken = isolated_detoken[:intra_token_index + 1] + new_kanji + isolated_detoken[1 + intra_token_index:]
                    if self.logging:
                        print(f'INSERT: New detoken: {new_detoken}')
                    detokens[index] = new_detoken
                    err_label[index] = 2
                    return detokens, err_label
