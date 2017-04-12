#! /usr/bin/python

__author__="Xinyu Li"
__date__ ="$Apr 10, 2017"


from hidden_markov_model import HiddenMarkovModel
from preprocess import dev_rare_unseen_iterator


class SimpleGeneTagger(HiddenMarkovModel):
    """
    A simple gene tagger that always produces the tag y* = argmax e(x|y) for
    each word x.
    """
    def __init__(self, n=3):
        super(SimpleGeneTagger, self).__init__(n)

    def tag(self, counts_file, dev_file, tag_file):
        """
        Tag the words according to the maximum emission parameters.
        """
        super(SimpleGeneTagger, self).read_counts(counts_file)
        dev_rare_unseen_iter = dev_rare_unseen_iterator(dev_file, self.get_all_words(),
                                                        self.get_rare_words())
        for word, tag in dev_rare_unseen_iter:
            if word:
                # Get all possible emission parameters for word
                emission_params = [self.emission_parameters[pair] for pair in self.emission_parameters
                                   if pair[0] == tag]
                # Get all possible emission (word, ne_tag) pairs for word
                emission_pairs = [pair for pair in self.emission_parameters if pair[0] == tag]
                # Find the index of the maximum emission parameter e(x|y)
                max_idx = emission_params.index(max(emission_params))
                # Return the ne_tag corresponding to maximum emission parameter e(x|y)
                max_ne_tag = emission_pairs[max_idx][1]
                tag_file.write("%s %s\n" % (word, max_ne_tag))
            else:
                tag_file.write("\n")
