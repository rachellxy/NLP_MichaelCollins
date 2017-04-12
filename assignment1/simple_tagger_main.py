#! /usr/bin/python

__author__ = "Xinyu Li"
__date__ = "$Apr 10, 2017"


from preprocess import process_train_rare_words
from simple_gene_tagger import SimpleGeneTagger
from eval_gene_tagger import corpus_iterator
from eval_gene_tagger import Evaluator


def train(train_input, train_output, counts_file):
    print("1. Training HMM...")
    simple_tagger = SimpleGeneTagger(3)
    simple_tagger.train(file(train_input))
    rare_words = simple_tagger.get_rare_words()
    print("2. Replace rare words...")
    process_train_rare_words(file(train_input), file(train_output, "w"), rare_words)
    print("3. Train HMM again with processed training data...")
    simple_tagger = SimpleGeneTagger(3)
    simple_tagger.train(file(train_output))
    simple_tagger.write_counts(file(counts_file, "w"))


def tag(counts_file, dev_input, tag_output):
    print("4. Tagging test file...")
    simple_tagger = SimpleGeneTagger(3)
    simple_tagger.tag(file(counts_file), file(dev_input), file(tag_output, "w"))


def evaluate(dev_key, tag_output):
    gs_iterator = corpus_iterator(file(dev_key))
    pred_iterator = corpus_iterator(file(tag_output), with_logprob=False)
    evaluator = Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    evaluator.print_scores()


train_input = "gene.train"
train_output = "gene_rare_processed.train"
counts_file = "count.txt"
dev_input = "gene.dev"
tag_output = "gene_dev.p1.out"
dev_key = "gene.key"

train(train_input, train_output, counts_file)
tag(counts_file, dev_input, tag_output)
evaluate(dev_key, tag_output)
