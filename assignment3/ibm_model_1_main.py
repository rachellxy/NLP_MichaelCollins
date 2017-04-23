#! /usr/bin/python
# -*- coding: utf-8 -*-

from ibm_model_1 import IBMModel1
from eval_alignment import main


ibmModel1 = IBMModel1()
ibmModel1.train("corpus.en", "corpus.es", "t_parameters.out")
ibmModel1.alignment("dev.en", "dev.es", "dev.out")
main(file("dev.key"), file("dev.out"))
