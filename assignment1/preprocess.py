#! /usr/bin/python

__author__ = "Xinyu Li"
__date__ = "$Apr 10, 2017"


from count_freqs import simple_conll_corpus_iterator
import sys

RARE_COUNT = 5
RARE = "_RARE_"
NUM = "_NUM_"
ALL_CAP = "_ALL_CAP_"
LAST_CAP = "_LAST_CAP_"


def process_train_rare_words(corpus_file, processed_file, rare_words, process_type="RARE"):
    """
    type = RARE: replace rare words (count < RARE_COUNT) with RARE in training data.
    type = FOUR: replace rare words (count < RARE_COUNT) with corresponding four word
    classes in training data.

    Four word classes:
    Numeric: The word is rare and contains at least one numeric characters.
    All Capitals: The word is rare and consists entirely of capitalized letters.
    Last Capital: The word is rare, not all capitals, and ends with a capital letter.
    Rare: The word is rare and does not fit in the other classes.
    """
    corpus_iter = simple_conll_corpus_iterator(corpus_file)
    for word, ne_tag in corpus_iter:
        if word:
            if word in rare_words:
                if process_type == "RARE":
                    processed_file.write("%s %s\n" % (RARE, ne_tag))
                elif process_type == "FOUR":
                    if any([str(i) in word for i in xrange(10)]):
                        processed_file.write("%s %s\n" % (NUM, ne_tag))
                    elif all([c.isupper() for c in word]):
                        processed_file.write("%s %s\n" % (ALL_CAP, ne_tag))
                    elif word[len(word) - 1].isupper() and any([not c.isupper() for c in word[:len(word) - 1]]):
                        processed_file.write("%s %s\n" % (LAST_CAP, ne_tag))
                    else:
                        processed_file.write("%s %s\n" % (RARE, ne_tag))
            else:
                processed_file.write("%s %s\n" % (word, ne_tag))
        else:
            processed_file.write("\n")


def dev_iterator(dev_file):
    """
    Iterate words in development data.
    """
    l = dev_file.readline()
    while l:
        word = l.strip()
        if word:  # If this is not an empty line
            yield word
        else:
            yield None
        l = dev_file.readline()


def dev_rare_unseen_iterator(dev_file, all_words, rare_words, process_type="RARE"):
    """
    Iterate words in development data and mark rare and unseen words.
    """
    dev_iter = dev_iterator(dev_file)
    for word in dev_iter:
        if word:
            # If the word is a rare word or unseen word in training corpus
            if (word in rare_words) or (word not in all_words):
                if process_type == "RARE":
                    yield word, RARE
                elif process_type == "FOUR":
                    if any([str(i) in word for i in xrange(10)]):
                        yield word, NUM
                    elif all([c.isupper() for c in word]):
                        yield word, ALL_CAP
                    elif word[len(word) - 1].isupper() and any([not c.isupper() for c in word[:len(word) - 1]]):
                        yield word, LAST_CAP
                    else:
                        yield word, RARE
            else:
                yield word, word
        else:
            yield (None, None)


def process_dev_rare_unseen_words(dev_file, processed_file, all_words, rare_words, process_type="RARE"):
    """
    Replace rare words (count < RARE_COUNT) and unseen words in training
    data with RARE in development data.
    """
    dev_rare_unseen_iter = dev_rare_unseen_iterator(dev_file, all_words, rare_words, process_type)
    for word, tag in dev_rare_unseen_iter:
        if tag:
            processed_file.write("%s\n" % tag)
        else:
            processed_file.write("\n")


def dev_sentence_rare_unseen_iterator(dev_file, all_words, rare_words, process_type="RARE"):
    """
    Iterate sentences in development data.
    Replace rare and unseen words with RARE.
    """
    dev_rare_unseen_iter = dev_rare_unseen_iterator(dev_file, all_words, rare_words, process_type)
    current_sentence = []
    current_sentence_processed = []
    for word, word_processed in dev_rare_unseen_iter:
        if word:
            current_sentence.append(word)
            current_sentence_processed.append(word_processed)
        else:
            if current_sentence:
                yield current_sentence, current_sentence_processed
                current_sentence = []
                current_sentence_processed = []
            else:  # Got empty input stream
                sys.stderr.write("WARNING: Got empty input file/stream.\n")
                raise StopIteration
    if current_sentence:
        yield current_sentence, current_sentence_processed
