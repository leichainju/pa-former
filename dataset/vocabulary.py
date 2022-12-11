""" vocabulary for src and tgt
adopted from NeuralCodeSum
"""

import unicodedata
from collections import Counter
from collections.abc import Iterable

PAD, PAD_WORD = 0, '<blank>'
UNK, UNK_WORD = 1, '<unk>'
BOS, BOS_WORD = 2, '<s>'
EOS, EOS_WORD = 3, '</s>'


def build_vocab(examples, fields, dict_size=None, no_special_token=False):
    words = load_words(examples, fields, dict_size)
    vocab = Vocabulary(words, no_special_token)
    return vocab


def load_words(examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in examples:
        for field in fields:
            _insert(ex[field])

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    return set(word for word, _ in most_common)


class Vocabulary:
    def __init__(self, words=None, no_special_token=False):
        if no_special_token:
            self.tok2ind = {PAD_WORD: PAD,
                            UNK_WORD: UNK}
            self.ind2tok = {PAD: PAD_WORD,
                            UNK: UNK_WORD}
        else:
            self.tok2ind = {PAD_WORD: PAD,
                            UNK_WORD: UNK,
                            BOS_WORD: BOS,
                            EOS_WORD: EOS}
            self.ind2tok = {PAD: PAD_WORD,
                            UNK: UNK_WORD,
                            BOS: BOS_WORD,
                            EOS: EOS_WORD}

        if words is not None:
            self.add_words(words)

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self.ind2tok
        elif isinstance(key, str):
            return self.normalize(key) in self.tok2ind
        else:
            raise RuntimeError('Invalid key type.')

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ind2tok.get(key, UNK_WORD)
        elif isinstance(key, str):
            return self.tok2ind.get(self.normalize(key), self.tok2ind.get(UNK_WORD))
        else:
            raise RuntimeError('Invalid key type.')

    def __setitem__(self, key, item):
        if isinstance(key, int) and isinstance(item, str):
            self.ind2tok[key] = item
        elif isinstance(key, str) and isinstance(item, int):
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, word):
        word = self.normalize(word)
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word

    def add_words(self, words):
        assert isinstance(words, Iterable)
        for word in words:
            self.add(word)

    def tokens(self):
        """ Get dictionary tokens. """
        tokens = [k for k in self.tok2ind if k not in {PAD_WORD, UNK_WORD}]
        return tokens

    def remove(self, key):
        if key in self.tok2ind:
            ind = self.tok2ind[key]
            del self.tok2ind[key]
            del self.ind2tok[ind]
            return True
        return False
