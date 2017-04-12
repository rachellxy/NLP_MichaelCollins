#! /usr/bin/python

__author__="Xinyu Li"
__date__ ="$Apr 10, 2017"


from collections import defaultdict
from count_freqs import Hmm
from preprocess import RARE_COUNT


class HiddenMarkovModel(Hmm):
    """
    Extend base Hmm class provided by assignment and add functionality
    to calculate emission / transition probability.
    """

    def __init__(self, n=3):
        # Initializing Hmm
        super(HiddenMarkovModel, self).__init__(n)
        # Word count
        self.word_cnt = defaultdict(int)
        # All words
        self.all_words = set()
        # Rare words
        self.rare_words = set()
        # Emission parameters e(x|y)
        self.emission_parameters = defaultdict(float)
        # 3-grams transition parameters q(yi|yi-2,yi-1)
        self.transition_parameters = defaultdict(float)

    def train(self, corpus_file):
        """
        Train the transition and emission parameters for HMM.
        """
        super(HiddenMarkovModel, self).train(corpus_file)
        self.calculate_word_count()
        self.calculate_rare_words()
        self.calculate_emission_parameters()

    def calculate_word_count(self):
        """
        Get word frequency of training corpus.
        """
        for word, ne_tag in self.emission_counts:
            self.word_cnt[word] += self.emission_counts[(word, ne_tag)]
            self.all_words.add(word)

    def get_word_count(self):
        return self.all_words

    def get_all_words(self):
        return self.all_words

    def calculate_rare_words(self):
        """
        Get rare words (count < RARE_COUNT) of training corpus.
        """
        for word in self.word_cnt:
            if self.word_cnt[word] < RARE_COUNT:
                self.rare_words.add(word)

    def get_rare_words(self):
        return self.rare_words

    def calculate_emission_parameters(self):
        """
        Calculate the emission parameters e(x|y) for each (word, ne_tag) pair.
        e(x|y) = Count(y->x)/Count(y)
        """
        for pair in self.emission_counts:
            self.emission_parameters[pair] = float(self.emission_counts[pair]) \
                                             / float(self.ngram_counts[0][tuple(pair[-1:])])

    def calculate_transition_parameters(self):
        """
        Calculate the transition parameters q(yi|yi-2,yi-1) for ne_tag 3-grams.
        q(yi|yi-2,yi-1) = Count(yi-2,yi-1,yi)/Count(yi-2,yi-1)
        """
        for trigram in self.ngram_counts[2]:
            self.transition_parameters[trigram] = float(self.ngram_counts[2][trigram]) \
                                                  / float(self.ngram_counts[1][tuple(trigram[:2])])

    def read_counts(self, corpusfile):
        """
        Read in the parameters from input file and re-calculate other parameters.
        """
        super(HiddenMarkovModel, self).read_counts(corpusfile)
        self.word_cnt = defaultdict(int)
        self.all_words = set()
        self.rare_words = set()
        self.emission_parameters = defaultdict(float)
        self.transition_parameters = defaultdict(float)
        self.calculate_word_count()
        self.calculate_rare_words()
        self.calculate_emission_parameters()
        self.calculate_transition_parameters()
