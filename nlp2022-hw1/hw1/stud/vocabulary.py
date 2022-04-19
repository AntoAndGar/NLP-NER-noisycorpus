from collections import Counter
from typing import Any, List

class Vocabulary():
    '''
    Vocabulary class

    Parameters:
    tokens: list of lists of tokens
    min_freq: minimum frequency of a word to be included in the vocabulary
    unk_token: token to be used for unknown words
    verbose: print info about the vocabulary size
    '''
    def __init__(self, tokens, min_freq = 1, unk_token = '<UNK>', verbose = False):
        super().__init__()
        self.verbose = verbose
        self.tokens = tokens
        self.min_freq = min_freq
        self.unk_tok = unk_token
        self.vocab = self.build_vocab(self.tokens, self.min_freq, self.unk_tok)

    def __len__(self):
        return len(self.vocab.keys())

    def build_vocab(self, tokens, min_freq, unk_tok):
        # build vocabulary
        # create a counter for frequencies of words
        words_counter = Counter()
        # count freq in trainig dataset 
        if type(tokens[0]) == list:
            words_counter.update([elem for sublist in tokens for elem in sublist])
        else:
            words_counter.update([elem for elem in tokens])
        #print(words_counter.most_common(100))

        # select only word with frequencies major of pre-selected minimum frequency
        words_counter = [ele for ele in words_counter if words_counter[ele] >= min_freq]

        vocabulary = {}
        # For each words
        for word in words_counter:
            if word not in vocabulary:  # if word has not been assigned an index yet
                vocabulary[word] = len(vocabulary.keys())  # Assign each word with a unique index

        # add Unk to the vocab
        if unk_tok not in vocabulary:
            vocabulary[unk_tok] = len(vocabulary.keys())
        #print(vocabulary)
            
        return vocabulary

    @staticmethod
    def word_to_index(tokens: List[str], vocab):
        """
        Given a list of tokens and a vocabulary returns the corresponding 
        list of indexes of the words/tokens in that vocabulary
        """
        index = []
        for w in tokens :
            if w in vocab.vocab:
                index.append(vocab.vocab[w])
            else:
                index.append(vocab.vocab[vocab.unk_tok])
        return index
        #return [vocab[w] for w in tokens if w in vocab ]

    @staticmethod
    def index_to_words(indexes, vocab):
        """
        Given a list of indexes and a vocabulary returns the corresponding 
        list of tokens in that vocabulary
        """
        inv_map = {value: key for key, value in vocab.vocab.items()}
        tokens = []
        for i in indexes :
            if i in inv_map:
                tokens.append(inv_map[i])
            else:
                tokens.append(vocab.unk_tok)
        return tokens

    @staticmethod
    def tag_to_index(tags: List[str], tag_dictionary):
        """
        Given a list of tags and a dict returns the corresponding 
        list of indexes of the tags in that dict
        """
        return [tag_dictionary[t] for t in tags]

    @staticmethod
    def index_to_tags(indexes, tag_dictionary):
        """
        Given a list of tags and a dict returns the corresponding 
        list of indexes of the tags in that dict
        """
        inv_map = {value: key for key, value in tag_dictionary.items()}
        return [inv_map[int(i)] for i in indexes]
