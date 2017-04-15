#! /usr/bin/python

__author__ = "Xinyu Li"
__date__ = "$Apr 13, 2017"


from CKYParser import CKYParser
from preprocess import process_train_rare_words
import eval_parser


def train(train_input, train_output, counts_file):
    print("1. Training PCFG parser...")
    ckyParser = CKYParser()
    ckyParser.train(train_input)
    rare_words = ckyParser.get_rare_words()
    print("2. Replacing rare words...")
    process_train_rare_words(train_input, train_output, rare_words)
    print("3. Writing counts file...")
    ckyParser = CKYParser()
    ckyParser.train(train_output)
    ckyParser.write_counts(counts_file)


def parse(counts_file, parse_file, parse_output):
    ckyParser = CKYParser()
    print("4. Reading counts file...")
    ckyParser.read_counts(counts_file)
    print("5. Parsing dev file...")
    ckyParser.parse(parse_file, parse_output)


def evaluate(dev_key, parse_output):
    eval_parser.main(dev_key, parse_output)


# train_input = "parse_train.dat"
train_input = "parse_train_vert.dat"
# train_output = "parse_train_processed.dat"
train_output = "parse_train_vert_processed.dat"
# counts_file = "parse_train.counts.out"
counts_file = "parse_train_vert.counts.out"
parse_file = "parse_dev.dat"
parse_output = "parse_dev.out"
dev_key = "parse_dev.key"

train(train_input, train_output, counts_file)
parse(counts_file, parse_file, parse_output)
evaluate(dev_key, parse_output)
