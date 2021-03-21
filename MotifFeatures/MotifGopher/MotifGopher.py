import datetime
import random
import os.path
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import time

from Utils.algs import memo

class MotifGopher:
    """I wish to pick out common motifs from an (almost) arbitrarily
    long sequence of raw texts, and I wish to finish in a reasonable
    amount of time. I also wish to be able to make precise statistical
    statements about the output of my algorithm. This class represents
    an effort to accomplish these objectives.
    """
    def __init__(
            self, texts,
            thresh=0.1,
            confidence=0.99,
            r=0.75,
            saveto='./'):
        """Initializes the MotifGopher with the variables that determine
        its behavior. These parameters ought not to change for a given
        MotifGopher instance.
        TEXTS - The texts to be searched for motifs.
        THRESH - The minimum mean number of times a motif provides _new_
            information per citation. If a motif provides no
            information, then it does not provide _new_ information. If
            it provides information that _one_ single other, _longer_
            motif provides, then it does not provide _new_ information.
            Else, it provides new information!
        CONFIDENCE - The required confidence level for any statistics
                that affect decisions made by the MotifGopher.
        R - The probability that a subsequent character will be included
                in a candidate motif.
        SAVETO - The location where logs and pickled output should be
                saved.
        """
        assert os.path.isfile(os.path.join(saveto, '.gopherlogs'))
        # Texts are padded with newlines to mark beginnings and endings.
        # These newlines are presented in lieu of start and end tags,
        # and they have the advantage of being only one character in
        # length, which may simplify the behavior of the MotifGopher as
        # it searches for motifs.
        self._texts = ['\n' + text + '\n' for text in texts]
        self._thresh = thresh
        self._confidence = confidence
        self._r = r
        self._saveto = saveto
        self.found_motifs = set()
        self._rejected_motifs = set()
        self._name = datetime.datetime.now().strftime(
            'Gopher_%I-%M%p_%d_%b_%Y'
        )
        self.log(str(self))
    def _motif(self):
        """Returns a randomly selected substring from the MotifGopher's
        private corpus.
        """
        selectedText = random.choice(self._texts)
        startIdx = random.randrange(0, len(selectedText))
        endIdx = startIdx + 1
        while random.random() < self._r and endIdx <= len(selectedText):
            endIdx += 1
        return selectedText[startIdx:endIdx]
    def hunt(self, sample_size=20, r=10):
        """Builds one motif and adds it to the list of found motifs if
        it meets the inclusion criteria (above threshold frequency, not
        already included). Returns the found motif if the inclusion
        criteria were met; otherwise, does nothing and returns None.
        SAMPLE_SIZE - The initial sample size used to determine the
            frequency of the found motif.
        R - The amount by which the sample size grows for each failure
            to ascertain whether the found motif is frequent enough.
        Promise: The values of the parameters SAMPLE_SIZE and R may
            affect performance, but they should not appreciably affect
            program output.
        """
        found = self._motif()
        if      (
                    found in self.found_motifs
                    or found in self._rejected_motifs
                    or not self._provides_information(found)
                ):
            self._rejected_motifs.add(found)
            return None
        to_be_replaced = None
        for other in self.found_motifs:
            if not self._provides_information(found, other):
                if len(other) > len(found):
                    self._rejected_motifs.add(found)
                    return None
                else:
                    # Note: Ties are broken arbitrarily, depending on
                    # set iteration order. No promises are made about
                    # which motif should be kept if two motifs have the
                    # same length.
                    to_be_replaced = other
                    break
        if to_be_replaced is not None:
            self.found_motifs.remove(to_be_replaced)
            self._rejected_motifs.add(to_be_replaced)
        self.found_motifs.add(found)
        return found
    def _get_motif_vec(self, motif, texts=None):
        """Returns a logical ndarray representing the indices of matches
        to MOTIF in a concatenated group of texts.

        If TEXTS is not set to None, uses the concatenated group of
        TEXTS; else, uses all texts.
        """
        if texts is None:
            if not hasattr(self, '_concatenated'):
                self._concatenated = ''.join(self._texts)
            concatenated = self._concatenated
        else:
            concatenated = ''.join(texts)
        ret = np.zeros(len(concatenated))
        idx = concatenated.find(motif)
        while idx != -1:
            ret[idx] = 1
            idx = concatenated.find(motif, idx + 1)
        return ret
    def _provides_information(self, motif0, motif1=None, sample_size=100, r=10):
        """Returns whether MOTIF provides information at average of at
        least the threshold number of times per citation.
        MOTIF0 - the motif that is to be assessed for the information it
            provides
        MOTIF1 - (optional) the motif against which MOTIF0 is to be
            compared for redundancy
        """
        p = 1
        provides_info = False
        while p > (1 - self._confidence):
            if sample_size < len(self._texts):
                texts = random.choices(self._texts, k=sample_size)
            else:
                texts = self._texts
            vec = self._get_motif_vec(motif0, texts)
            if motif1 is None:
                successes = np.sum(vec)
            else:
                successes = np.sum(vec != self._get_motif_vec(motif1, texts))
            char_thresh = self._get_thresh_by_char(texts)
            provides_info = successes / len(vec) >= char_thresh
            p = stats.binom_test(
                successes,
                n=len(vec),
                p=char_thresh
            ) if texts is not self._texts else 0 # p is meaningless if sample=population
            sample_size *= r
        # DEBUG
        if not provides_info and motif1 is not None:
            print('DEBUG: "{}" too similar to "{}"'.format(motif0, motif1))
        return provides_info
    def _get_thresh_by_char(self, texts):
        """Returns the minimum number of characters required per text at
        which new information must be provided in order to meet the
        threshold stipulated in the constructor. Note: This value is
        imprecise, and the smaller the TEXTS list is, the
        more imprecise the output value will be.
        """
        return self._thresh / np.average([len(text) for text in texts])
    @memo
    def _max_motif_len(self):
        """Returns the length of the longest found motif."""
        return max(len(motif) for motif in self.found_motifs)

    def log(self, message):
        """Appends MESSAGE to the file .gopherlogs, prefixed with the
        gopher's name.
        """
        with open(os.path.join(self._saveto, '.gopherlogs'), 'a') as f:
            f.write('{}: {}\n'.format(self._name, message))
    def save(self, suffix=''):
        """Saves a dictionary of found motifs and their frequencies to a
        pickle file.
        """
        path = os.path.join(
            self._saveto, self._name + '_motifs' + suffix + '.pickle')
        with open(path, 'wb') as dbfile:
            pickle.dump(self.found_motifs, dbfile, pickle.HIGHEST_PROTOCOL)
    def plot_hunt(self, n=25000, resolution=50):
        """Plots the growth of the number of found motifs relative to
        the number of hunts.
        """
        hunts = []
        num_found = []
        t0 = time.time()
        for i in range(n):
            if i % 100 == 0:
                print('Hang in there. {:.2f}% finished after {:.2f} '
                      'seconds. {} motifs checked and {} motifs found.'.format(
                    100 * i / n,
                    time.time() - t0,
                    i,
                    len(self.found_motifs)
                ))
            if i % resolution == 0:
                hunts.append(i)
                num_found.append(len(self.found_motifs))
            self.hunt()
        plt.plot(hunts, num_found)
        plt.show()
    def run(self, verbose=True):
        """Manages a sequence of operations, with progress reports and
        requests for human input. Manages logging and saving.
        """
        last_nfound = len(self.found_motifs)
        while True:
            n = int(input('How many hunt operations would you like to run? '
                          '(at least 25K recommended.) '))
            self.plot_hunt(n=n)
            new_nfound = len(self.found_motifs)
            message = (
                'After {} hunt operations, the number of found motifs has\n'
                'increased by {} to {}.'.format(
                    n,
                    new_nfound - last_nfound,
                    new_nfound
                )
            )
            self.log(message)
            print(message)
            last_nfound = new_nfound
            if 'y' in input('Would you like to save and stop hunting for '
                            'motifs? ').lower():
                break
        self.save()
    def __str__(self):
        return ('MotifGopher instance with thresh={}, '
                'confidence={}, r={}'.format(
                    self._thresh, self._confidence, self._r
        ))

def cross_corr(arr1, arr2, padding):
    """Returns the max cross correlation of ARR1 and ARR2, with a
    displacement of magnitude in the interval [0, PADDING]."""
    return max(np.correlate(np.pad(arr1, padding), arr2))