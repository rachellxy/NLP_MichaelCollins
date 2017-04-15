#! /usr/bin/python

__author__ = "Xinyu Li"
__date__ = "$Apr 14, 2017"


from collections import defaultdict
from math import log
from math import isinf
from pcfg import PCFG
from preprocess import RARE
import json


S = "SBARQ"


class CKYParser(PCFG):

    def __init__(self):
        super(CKYParser, self).__init__()

    def train(self, parse_file):
        """
        Train the parameters of the PCFG parser.
        """
        for l in open(parse_file):
            t = json.loads(l)
            self.count(t)
        self.calculate_word_count()
        self.calculate_rare_words()
        self.calculate_unary_parameters()
        self.calculate_binary_parameters()

    def parse(self, parse_file, parse_output):
        """
        Parse the sentences in development data.
        """
        output = file(parse_output, "w")
        for sentence in open(parse_file):
            json.dump(self.CKY(sentence), output)
            output.write("\n")

    def CKY(self, sentence):
        """
        Implement CKY algorithm to compute the parse tree using argmax p(t) for given sentence.
        """
        words = sentence.strip().split(" ")
        n = len(words)
        # Replace rare words or unseen words with _RARE_
        for i in range(0, n):
            if (words[i] in self.rare_words) or (words[i] not in self.all_words):
                words[i] = RARE

        pi = defaultdict(float)
        bp_binary = defaultdict(str)  # back pointer for binary rules X -> Y1Y2
        bp_s = defaultdict(int)  # back pointer for s = i...(j-1)

        for i in range(1, n + 1):  # for i = 1...n
            w = words[i - 1]
            for x in self.nonterm:  # for all X in non-terminal symbols
                if (x, w) in self.unary:  # if X -> wi is in unary rules
                    # pi(i,i,X) = q(X -> wi)
                    pi[(i, i, x)] = log(self.unary_parameters[(x, w)], 2)
                else:
                    # otherwise, pi(i,i,X) = 0
                    pi[(i, i, x)] = float("-Inf")

        x_binary = {}
        for x in self.nonterm:
            for (xx, y1, y2) in self.binary:
                if xx == x:
                    if x not in x_binary:
                        x_binary[x] = []
                    x_binary[x].append((xx, y1, y2))

        for l in range(1, n):  # for l = 1...(n-1), indicating the interval between i and j
            for i in range(1, n - l + 1):  # for i = 1...(n-l)
                j = i + l  # set j = i + l
                for x in x_binary:
                    pi_max = float("-Inf")
                    bp_binary_max = None
                    bp_s_max = 0
                    for (xx, y1, y2) in x_binary[x]:
                        for s in range(i, j):
                            if ((i, s, y1) not in pi) or isinf(pi[(i, s, y1)]) or \
                                    ((s+1, j, y2) not in pi) or isinf(pi[(s+1, j, y2)]):
                                continue
                            tmp = log(self.binary_parameters[(xx, y1, y2)], 2) + pi[(i, s, y1)] + \
                                  pi[(s+1, j, y2)]
                            if tmp > pi_max:
                                pi_max = tmp
                                bp_binary_max = (xx, y1, y2)
                                bp_s_max = s
                    if not isinf(pi_max):
                        pi[(i, j, x)] = pi_max  # pi(i,j,X) = max q(X -> Y1Y2)*pi(i,s,Y1)*pi(s+1,j,Y2)
                        bp_binary[(i, j, x)] = bp_binary_max
                        bp_s[(i, j, x)] = bp_s_max

        root = None  # Find the root of parse tree
        if (1, n, S) in pi:
            root = S
        else:
            pi_max = float("-Inf")
            for (i, j, x) in pi:
                if i == 1 and j == n:
                    if pi[(i, j, x)] > pi_max:
                        pi_max = pi[(i, j, x)]
                        root = x

        parse_tree = self.generate_tree(sentence, 1, n, root, bp_binary, bp_s)
        return parse_tree

    def generate_tree(self, sentence, i, j, x, bp_binary, bp_s):
        """
        Generate parse tree for one sentence.
        """
        words = sentence.strip().split(" ")
        parse_tree = list()
        parse_tree.append(x)
        if i == j:
            parse_tree.append(words[i - 1])
        else:
            (xx, y1, y2) = bp_binary[(i, j, x)]
            s = bp_s[(i, j, x)]
            parse_tree.append(self.generate_tree(sentence, i, s, y1, bp_binary, bp_s))
            parse_tree.append(self.generate_tree(sentence, s+1, j, y2, bp_binary, bp_s))
        return parse_tree
