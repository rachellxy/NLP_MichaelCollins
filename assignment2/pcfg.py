#! /usr/bin/python

__author__ = "Xinyu Li"
__date__ = "$Apr 14, 2017"


from count_cfg_freq import Counts
from collections import defaultdict
from preprocess import RARE_COUNT


class PCFG(Counts):

    def __init__(self):
        super(PCFG, self).__init__()
        # Word (terminal symbols) count
        self.word_cnt = defaultdict(int)
        # All words (terminal symbols)
        self.all_words = set()
        # Rare words (terminal symbols)
        self.rare_words = set()
        # Unary rule parameters q(X -> w)
        self.unary_parameters = defaultdict(float)
        # Binary rule parameters q(X -> Y1Y2)
        self.binary_parameters = defaultdict(float)

    def calculate_word_count(self):
        """
        Get all words (terminal symbols) in training data.
        """
        for (x, w) in self.unary:
            self.word_cnt[w] += self.unary[(x, w)]
            self.all_words.add(w)

    def calculate_rare_words(self):
        """
        Get rare words (count < RARE_COUNT) in training data.
        """
        for word in self.word_cnt:
            if self.word_cnt[word] < RARE_COUNT:
                self.rare_words.add(word)

    def get_rare_words(self):
        return self.rare_words

    def calculate_unary_parameters(self):
        """
        Calculate unary rule parameters q(X -> w) = Count(X -> w) / Count(X).
        """
        for (x, w) in self.unary:
            self.unary_parameters[(x, w)] = float(self.unary[(x, w)]) / float(self.nonterm[x])

    def calculate_binary_parameters(self):
        """
        Calculate binary rule parameters q(X -> Y1Y2) = Count(X -> Y1Y2) / Count(X).
        """
        for (x, y1, y2) in self.binary:
            self.binary_parameters[(x, y1, y2)] = float(self.binary[(x, y1, y2)]) / float(self.nonterm[x])

    def write_counts(self, counts_file):
        """
        Write counts of non-terminal symbols, unary rules and binary rules to output file.
        """
        output = file(counts_file, "w")

        for symbol, count in self.nonterm.iteritems():
            output.write("%i %s %s\n" % (count, "NONTERMINAL", symbol))

        for (sym, word), count in self.unary.iteritems():
            output.write("%i %s %s %s\n" % (count, "UNARYRULE", sym, word))

        for (sym, y1, y2), count in self.binary.iteritems():
            output.write("%i %s %s %s %s\n" % (count, "BINARYRULE", sym, y1, y2))

    def read_counts(self, counts_file):
        """
        Read in the parameters from input file and re-calculate the parameters.
        """
        self.nonterm = {}
        self.unary = {}
        self.binary = {}
        self.word_cnt = defaultdict(int)
        self.all_words = set()
        self.rare_words = set()
        self.unary_parameters = defaultdict(float)
        self.binary_parameters = defaultdict(float)

        counts = file(counts_file)
        l = counts.readline()
        while l:
            line = l.strip()
            if line:
                parts = line.split(" ")
                cnt = int(parts[0])
                if parts[1] == "NONTERMINAL":
                    symbol = parts[2]
                    self.nonterm.setdefault(symbol, 0)
                    self.nonterm[symbol] = cnt
                elif parts[1] == "UNARYRULE":
                    x, w = (parts[2], parts[3])
                    self.unary.setdefault((x, w), 0)
                    self.unary[(x, w)] = cnt
                elif parts[1] == "BINARYRULE":
                    x, y1, y2 = (parts[2], parts[3], parts[4])
                    self.binary.setdefault((x, y1, y2), 0)
                    self.binary[(x, y1, y2)] = cnt
            l = counts.readline()

        # Re-calculate the parameters
        self.calculate_word_count()
        self.calculate_rare_words()
        self.calculate_unary_parameters()
        self.calculate_binary_parameters()
