#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Xinyu Li"
__date__ = "$Apr 20, 2017"


from ibm_model_1 import IBMModel1
from ibm_model_1 import NULL
from collections import defaultdict


class IBMModel2(IBMModel1):
    """
    Estimate the alignment parameters q(j|i,l,m) and the translation
    parameters t(f|e) using IBM Model 2.
    """

    def __init__(self):
        super(IBMModel2, self).__init__()
        self.q_parameters = defaultdict()

    def train(self, e_file, f_file, t_file):
        """
        Train alignment parameters and translation parameters for IBM model 2.
        """
        self.read_file(e_file, f_file)
        self.init_q_parameters()
        self.init_t_parameters(t_file)
        self.em_ibm_model2()

    def alignment(self, e_dev, f_dev, a_file):
        """
        Find alignments for the development sentence pairs dev.en / dev.es.
        a_i = argmax_j q(j|i,l,m) * t(f_i|e_j)
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
            l = len(e_sentence) - 1  # eliminate NULL while calculating the length for English sentence
            f_sentence = dev_f_sentences[k-1]
            m = len(f_sentence)
            for i in range(1, m+1):
                f_i = f_sentence[i-1]
                q_t_max = float("-Inf")
                j_max = 0
                for j in range(0, l+1):
                    e_j = e_sentence[j]
                    if self.q_parameters[(i, l, m)][j] * self.t_parameters[e_j][f_i] > q_t_max:
                        q_t_max = self.q_parameters[(i, l, m)][j] * self.t_parameters[e_j][f_i]
                        j_max = j
                output.write("%i %i %i\n" % (k, j_max, i))

    def init_q_parameters(self):
        """
        Initialize q(j|i,l,m) = 1 / (l + 1).
        """
        for k in range(0, len(self.e_sentences)):
            m = len(self.f_sentences[k])
            l = len(self.e_sentences[k])
            for i in range(1, m+1):
                if (i, l, m) not in self.q_parameters:
                    self.q_parameters[(i, l, m)] = defaultdict(float)
                for j in range(0, l+1):  # including NULL as the 0th positive in English sentence
                    self.q_parameters[(i, l, m)][j] = float(1) / float(l + 1)

    def init_t_parameters(self, t_file):
        """
        Use the translation parameters trained by IBM Model 1.
        """
        self.read_t_parameters(t_file)

    def em_ibm_model2(self, T=5):
        """
        EM algorithm for IBM Model 2.
        """
        n = len(self.e_sentences)
        delta = defaultdict()

        for t in range(0, T):
            print "Iteration %i" % (t + 1)
            e_f_cnt = defaultdict(float)
            e_cnt = defaultdict(float)
            i_j_l_m_cnt = defaultdict(float)
            i_l_m_cnt = defaultdict(float)

            for k in range(1, n+1):
                e_sentence = [NULL]
                e_sentence.extend(self.e_sentences[k-1])
                l = len(e_sentence) - 1  # eliminate NULL while calculating the length for English sentence
                f_sentence = self.f_sentences[k-1]
                m = len(f_sentence)
                for i in range(1, m+1):
                    f_i = f_sentence[i-1]
                    for j in range(0, l+1):
                        e_j = e_sentence[j]
                        denominator = sum([self.q_parameters[(i, l, m)][j_k] * self.t_parameters[e_sentence[j_k]][f_i]
                                       for j_k in range(0, l+1)])
                        delta[(k, i, j)] = self.q_parameters[(i, l, m)][j] * self.t_parameters[e_j][f_i] / denominator
                        e_f_cnt[(e_j, f_i)] += delta[(k, i, j)]
                        e_cnt[e_j] += delta[(k, i, j)]
                        i_j_l_m_cnt[(i, j, l, m)] += delta[(k, i, j)]
                        i_l_m_cnt[(i, l, m)] += delta[(k, i, j)]

            for (i, j, l, m) in i_j_l_m_cnt:
                self.q_parameters[(i, l, m)][j] = i_j_l_m_cnt[(i, j, l, m)] / i_l_m_cnt[(i, l, m)]

            for (e, f) in e_f_cnt:
                self.t_parameters[e][f] = e_f_cnt[(e, f)] / e_cnt[e]
