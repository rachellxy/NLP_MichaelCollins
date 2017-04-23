#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Xinyu Li"
__date__ = "$Apr 22, 2017"


from collections import defaultdict
from ibm_model_1 import IBMModel1
from ibm_model_2 import IBMModel2
from eval_alignment import main


class PhraseTranslation(object):
    """
    Grow alignments using the intersection and union of alignments estimated
    from p(f|e) and p(e|f).
    """

    def __init__(self):
        self.f_e_alignments = defaultdict()
        self.e_f_alignments = defaultdict()
        self.alignments = []

    def read_init_alignments(self, f_e_file, e_f_file):
        """
        Read alignments estimated from p(f|e) and p(e|f) from files.
        """
        for l in open(f_e_file):
            line = l.strip()
            if line:
                parts = line.split(" ")
                k = int(parts[0])
                j = int(parts[1])
                i = int(parts[2])
                if k not in self.f_e_alignments:
                    self.f_e_alignments[k] = defaultdict(int)
                self.f_e_alignments[k][i] = j  # every foreign word is aligned with only one English word

        for l in open(e_f_file):
            line = l.strip()
            if line:
                parts = line.split(" ")
                k = int(parts[0])
                i = int(parts[1])
                j = int(parts[2])
                if k not in self.e_f_alignments:
                    self.e_f_alignments[k] = defaultdict(int)
                self.e_f_alignments[k][j] = i  # every English word is aligned with only one foreign word

    def intersection(self, inter_file):
        """
        Get the intersection of alignments estimated from p(f|e) and p(e|f).
        """
        output = file(inter_file, "w")
        for k in self.f_e_alignments:
            for i in self.f_e_alignments[k]:
                j = self.f_e_alignments[k][i]
                if self.e_f_alignments[k][j] == i:
                    self.alignments.append((k, j, i))
                    output.write("%i %i %i\n" % (k, j, i))


# Estimate IBM model 2 for p(f|e)
ibmModel1_f_e = IBMModel1()
ibmModel1_f_e.train("corpus.en", "corpus.es", "t_parameters_f_e.out")
ibmModel2_f_e = IBMModel2()
ibmModel2_f_e.train("corpus.en", "corpus.es", "t_parameters_f_e.out")
ibmModel2_f_e.alignment("dev.en", "dev.es", "dev_f_e.out")

# Estimate IBM model 2 for p(e|f)
ibmModel1_e_f = IBMModel1()
ibmModel1_e_f.train("corpus.es", "corpus.en", "t_parameters_e_f.out")
ibmModel2_e_f = IBMModel2()
ibmModel2_e_f.train("corpus.es", "corpus.en", "t_parameters_e_f.out")
ibmModel2_e_f.alignment("dev.es", "dev.en", "dev_e_f.out")

pt = PhraseTranslation()
pt.read_init_alignments("dev_f_e.out", "dev_e_f.out")
pt.intersection("dev_intersection.out")
main(file("dev.key"), file("dev_intersection.out"))
