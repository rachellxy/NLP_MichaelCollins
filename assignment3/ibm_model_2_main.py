#! /usr/bin/python
# -*- coding: utf-8 -*-

from ibm_model_2 import IBMModel2
from eval_alignment import main

ibmModel2 = IBMModel2()
ibmModel2.train("corpus.en", "corpus.es", "t_parameters.out")
ibmModel2.alignment("dev.en", "dev.es", "dev.out")
main(file("dev.key"), file("dev.out"))