#! /usr/bin/python

__author__ = "Xinyu Li"
__date__ = "$Apr 13, 2017"

import json

"""
Process the rare words in training and dev parse trees.
"""

RARE_COUNT = 5
RARE = "_RARE_"


def tree_iterator(tree, rare_words):
    """
    Replace infrequent words (Count(x) < 5) in the original parse tree
    with a common symbol _RARE_.
    """
    if isinstance(tree, basestring): return

    if len(tree) == 3:
        # It is a binary rule.
        tree_iterator(tree[1], rare_words)
        tree_iterator(tree[2], rare_words)
    elif len(tree) == 2:
        # It is a unary rule.
        if tree[1] in rare_words:  # Replace the rare words with _RARE_
            tree[1] = RARE

    return tree


def process_train_rare_words(parse_file, processed_file, rare_words):
    """
    Replace infrequent words (Count(x) < 5) in the original training data file
    with a common symbol _RARE_.
    """
    processed_output = file(processed_file, "w")
    for l in open(parse_file):
        if l.strip():
            t = json.loads(l)
            t = tree_iterator(t, rare_words)
            json.dump(t, processed_output)
            processed_output.write("\n")
