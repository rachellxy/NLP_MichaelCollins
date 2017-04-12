#! /usr/bin/python

__author__="Xinyu Li"
__date__ ="$Apr 11, 2017"


from collections import defaultdict
from hidden_markov_model import HiddenMarkovModel
from math import log
from preprocess import dev_sentence_rare_unseen_iterator

class ViterbiTagger(HiddenMarkovModel):
    """
    A tagger that Compute y1...yn using Viterbi algorithm with maximum likelihood
    estimates for transitions and emissions.
    """
    def __init__(self, n=3):
        super(ViterbiTagger, self).__init__(n)

    def tag(self, counts_file, dev_file, tag_file, process_type="RARE"):
        super(ViterbiTagger, self).read_counts(counts_file)
        dev_sentence_rare_unseen_iter = dev_sentence_rare_unseen_iterator(dev_file, self.get_all_words(),
                                                                          self.get_rare_words(), process_type)
        for sent, sent_processed in dev_sentence_rare_unseen_iter:
            tag_seq = self.sentence_tag(sent_processed)
            for i in xrange(len(sent)):
                tag_file.write("%s %s\n" % (sent[i], tag_seq[i]))
            tag_file.write("\n")

    def sentence_tag(self, sentence):
        """
        Tag a sentence x1...xn using transition parameters q(s|u, v) and emission
        parameters e(x|s).
        """
        pi = []  # pi(k, u, v)
        bp = []  # bp(k, u, v)
        n = len(sentence)
        for k in xrange(n + 1): # for k = 1...n
            pi.append(defaultdict(float))
            bp.append(defaultdict(str))
            if k == 0:
                pi[k][("*", "*")] = log(1.0, 2) # Use log to scale
                continue
            U = self.all_states
            V = self.all_states
            W = self.all_states
            x = sentence[k - 1]
            if k == 1:
                W = set(["*"])
                U = set(["*"])
            if k == 2:
                W = set(["*"])
            for u in U:
                for v in V:
                    pi_max = float("-Inf")
                    bp_max = None
                    for w in W:
                        if (w, u, v) not in self.transition_parameters:
                            continue
                        if (x, v) not in self.emission_parameters:
                            continue
                        pi_tmp = pi[k - 1][(w, u)] + \
                                 log(self.transition_parameters[(w, u, v)], 2) + \
                                 log(self.emission_parameters[(x, v)], 2)
                        if pi_tmp > pi_max:
                            pi_max = pi_tmp
                            bp_max = w
                    pi[k][(u, v)] = pi_max
                    bp[k][(u, v)] = bp_max
        # Get (yn-1, yn)
        U = self.all_states
        V = self.all_states
        if n == 0:
            U = set(["*"])
            V = set(["*"])
        if n == 1:
            U = set(["*"])
        pi_n_max = float("-Inf")
        bp_n_1_max = None
        bp_n_max = None
        for u in U:
            for v in V:
                if (u, v, "STOP") not in self.transition_parameters:
                    continue
                pi_tmp = pi[n][(u, v)] + \
                         log(self.transition_parameters[(u, v, "STOP")], 2)
                if pi_tmp > pi_n_max:
                    pi_n_max = pi_tmp
                    bp_n_1_max = u
                    bp_n_max = v
        # Unravel the backpointers to find the highest probability sequence
        seq = range(0, n + 1) # To keep the index consistent with bp
        seq[n - 1] = bp_n_1_max # yn-1
        seq[n] = bp_n_max # yn
        for k in xrange(n - 2, 0, -1):
            seq[k] = bp[k + 2][(seq[k + 1], seq[k + 2])]
        return seq[1:]
