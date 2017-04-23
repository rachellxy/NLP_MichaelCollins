#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Xinyu Li"
__date__ = "$Apr 17, 2017"


from collections import defaultdict


NULL = "_NULL_"


class IBMModel1(object):
    """
    Estimate the translation parameters t(f|e) from IBM model1.
    """

    def __init__(self):
        # English sentences
        self.e_sentences = list()
        # Foreign language sentences
        self.f_sentences = list()
        # Translation parameters
        self.t_parameters = defaultdict()

    def read_file(self, e_file, f_file):
        """
        Read in source language and target language sentences.
        Empty lines manually deleted from the training data.
        """
        for l in open(e_file):
            sentence = l.strip()
            if sentence:
                self.e_sentences.append(sentence.split(" "))

        for l in open(f_file):
            sentence = l.strip()
            if sentence:
                self.f_sentences.append(sentence.split(" "))

    def train(self, e_file, f_file, t_file):
        """
        Train translation parameters for IBM model 1.
        """
        self.read_file(e_file, f_file)
        self.init_t_parameters()
        self.em_ibm_model1()
        self.write_t_parameters(t_file)

    def alignment(self, e_dev, f_dev, a_file):
        """
        Find alignments for the development sentence pairs dev.en / dev.es.
        For each sentence, align each foreign word f_i to the English word with the highest t(f|e) score,
        a_i = argmax_j t(f_i|e_j)
        """
        output = file(a_file, "w")
        dev_e_sentences = []
        for l in open(e_dev):
            sentence = l.strip()
            if sentence:
                dev_e_sentences.append(sentence.split(" "))

        dev_f_sentences = []
        for l in open(f_dev):
            sentence = l.strip()
            if sentence:
                dev_f_sentences.append(sentence.split(" "))

        n = len(dev_e_sentences)
        for k in range(1, n+1):
            e_sentence = [NULL]
            e_sentence.extend(dev_e_sentences[k-1])
            l = len(e_sentence)
            f_sentence = dev_f_sentences[k-1]
            m = len(f_sentence)
            for i in range(1, m+1):
                f_i = f_sentence[i-1]
                t_max = float("-Inf")
                j_max = 0
                for j in range(0, l):
                    e_j = e_sentence[j]
                    if self.t_parameters[e_j][f_i] > t_max:
                        t_max = self.t_parameters[e_j][f_i]
                        j_max = j
                output.write("%i %i %i\n" % (k, j_max, i))

    def init_t_parameters(self):
        """
        Initialize translation parameters t(f|e) = 1/n(e).
        n(e) is the number of different words that occur in any translation
        of a sentence containing e.
        """
        self.t_parameters[NULL] = defaultdict(float)
        for i in range(len(self.e_sentences)):
            for e in self.e_sentences[i]:
                if e not in self.t_parameters:
                    self.t_parameters[e] = defaultdict(float)
                for f in self.f_sentences[i]:
                    # t_parameters[en_word] contains all foreign words that could align to en_word
                    self.t_parameters[e][f] += 1
            for f in self.f_sentences[i]:
                # NULL can be aligned to any foreign word
                self.t_parameters[NULL][f] += 1

        # Initialize t(f|e) = 1 / n(e), where n(e) is the number of different words
        # that occur in any translation of a sentence containing e
        for e in self.t_parameters:
            for f in self.t_parameters[e]:
                self.t_parameters[e][f] = float(1) / float(len(self.t_parameters[e]))

    def em_ibm_model1(self, T=5):
        """
        EM algorithm for IBM Model 1.
        """
        n = len(self.e_sentences)  # Number of sentences in the corpus
        delta = defaultdict()

        for t in range(0, T):  # For t = 1...T
            print "Iteration %i" % (t + 1)
            e_f_cnt = defaultdict(float)  # Count(e_j,f_i)
            e_cnt = defaultdict(float)  # Count(e_j)

            for k in range(1, n+1):  # For k = 1...n
                # Append special English word NULL to the 0th position of English sentence
                e_sentence = [NULL]
                e_sentence.extend(self.e_sentences[k-1])
                l = len(e_sentence)  # including NULL
                f_sentence = self.f_sentences[k-1]
                m = len(f_sentence)
                for i in range(1, m+1):  # For i = 1...m_k
                    f_i = f_sentence[i-1]
                    for j in range(0, l):  # For j = 0...l_k
                        e_j = e_sentence[j]
                        denominator = sum([self.t_parameters[e][f_i] for e in e_sentence])
                        delta[(k, i, j)] = self.t_parameters[e_j][f_i] / denominator
                        e_f_cnt[(e_j, f_i)] += delta[(k, i, j)]
                        e_cnt[e_j] += delta[(k, i, j)]

            # Update translation parameters
            for (e, f) in e_f_cnt:
                self.t_parameters[e][f] = e_f_cnt[(e, f)] / e_cnt[e]

    def write_t_parameters(self, t_file):
        """
        Write translation parameters to a file.
        """
        output = file(t_file, "w")
        for e in self.t_parameters:
            for f in self.t_parameters[e]:
                output.write("%s %s %f\n" % (e, f, self.t_parameters[e][f]))

    def read_t_parameters(self, t_file):
        """
        Read t parameters from file.
        """
        self.t_parameters = defaultdict()
        for l in open(t_file):
            line = l.strip()
            if line:
                parts = line.split(" ")
                e = parts[0]
                f = parts[1]
                t = float(parts[2])
                if e not in self.t_parameters:
                    self.t_parameters[e] = defaultdict(float)
                self.t_parameters[e][f] = t
