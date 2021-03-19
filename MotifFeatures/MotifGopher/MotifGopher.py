import datetime
import random
import os.path
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
from scipy import stats

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
        self.found_motifs = dict()
        self._rejected_motifs = dict()
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
        if found in self.found_motifs:
            return None
        while True:
            if (sample_size >= len(self._texts)):
                sample = self._texts
                frequency = np.mean([found in text for text in sample])
                if frequency > self._thresh:
                    self.found_motifs[found] = frequency
                    return found
                return None
            else:
                sample = random.choices(self._texts, k=sample_size)
                count = sum(found in text for text in sample)
                p = stats.binom_test(count, n=sample_size, p=self._thresh)
                if p < 1 - self._confidence:
                    frequency = count / len(sample)
                    if frequency > self._thresh:
                        self.found_motifs[found] = frequency
                        return found
                    return None
            sample_size *= r
    def _get_motif_vec(self, motif, texts=None):
        """Returns a logical ndarray representing the indices of matches
        to MOTIF in a concatenated group texts.

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
        idx = self._concatenated.find(motif)
        while idx != -1:
            ret[idx] = 1
            idx = self._concatenated.find(motif, idx + 1)
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
                successes = np.not_equal(vec, self._get_motif_vec(motif1))
            char_thresh = self._get_thresh_by_char(texts)
            provides_info = successes / len(vec) >= char_thresh
            p = stats.binom_test(
                successes,
                n=len(vec),
                p=char_thresh
            ) if texts != self._texts else 0 # p is meaningless if sample=population
            sample_size *= r
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
    @memo
    def _get_correlated(key, other_key, sample_size=100, r=10):
        """Returns whether the two keys have a max cross-correlation
        that exceeds MAX_CORR. Rate of any error should be approximately 
        less than (1 - CONFIDENCE) (because a two-tailed test is used,
        and sample size is increased by a LARGE factor until either the
        population is sampled or p-value is below (1 - CONFIDENCE)).

        TODO: Potentially do this in a statistically correct way. Here
        are two issues in the current implementation:
        * They take the p-value corresponding to the highest correlation
        from a series of multiple tests, so the p-value is no longer
        valid. (This problem is analogous to the problem of
        "p-hacking.")
        * The samples are modeled as independent samples of individual
        substring positions, but they obviously are not: Instead, they
        are cluster samples because substring positions are "clustered"
        by the text from which they came. This is a huge problem, but it
        is hard to fix it without making this algorithm unacceptably
        slow.
        * With the displacement associated with cross-correlation, the
        value of n is no longer well-defined.
        """
        p = 1
        excessively_correlated = False
        while p >= 1 - self._confidence:
            texts = random.choices(self._texts, k=sample_size)
            corr = correlation(
                arr1    = self._get_motif_vec(key, texts),
                arr2    = self._get_motif_vec(other_key, texts),
                padding = self._max_motif_len()
            )
            excessively_correlated = corr > self._max_corr
            if excessively_correlated:
                p = 2 * (1 - dist.cdf(corr))
            else:
                p = 2 * dist.cdf(corr)

    def purge(self, verbose=False):
        """Purges the found motifs of excessively correlated pairs;
        includes the longer motif whenever two motifs are excessively
        correlated. Ties will be broken rather arbitrarily (but
        deterministically!) by standard string comparison.
        It is recommended to call this method only once,
        after collecting a satisfactory number of motifs.

        WARNING: The implementation I have right now is what might be
        called the "naive implementation." Yes, I am naive: I'm
        nineteen. So, it's kind of slow, and it takes quadratic time wrt
        the number of found motifs.
        """
        corr_padding = self._max_motif_len()
        purged = dict()
        for i, key in enumerate(self.found_motifs.keys()):
            for other_key in self.found_motifs.keys():
                if key == other_key:
                    continue
                if len(key) > len(other_key):
                    continue
                if len(key) == len(other_key) and key > other_key:
                    continue
                corr = correlation(
                    self._get_motif_vec(key),
                    self._get_motif_vec(other_key),
                    corr_padding
                )
                if corr > self._max_corr:
                    if verbose:
                        print('Deleting "{}" because it was too similar to '
                              '"{}" (similarity: {:.4f})'
                            .format(key, other_key, corr))
                    break
            else:
                purged[key] = self.found_motifs[key]
            if verbose:
                if i % 10 == 0:
                    print('Searched {}% of the keys...'.format(
                        100 * i / len(self.found_motifs)
                    ))
        self.found_motifs = purged

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
    def plot_hunt(self, resolution=50, n=25000):
        """Plots the growth of the number of found motifs relative to
        the number of hunts.
        """
        hunts = []
        num_found = []
        for i in range(n):
            if i % 1000 == 0:
                print('Hang in there. {}% finished.'.format(100 * i / n))
            self.hunt()
            if i % resolution == 0:
                hunts.append(i)
                num_found.append(len(self.found_motifs))
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
        self.save(suffix='_unpurged')
        if 'y' in input('Would you like to purge motifs that are highly '
                        'correlated with other motifs of at least the same '
                        'length? '):
            self.purge(verbose)
            print('Saving... ', end='')
            self.save(suffix='_purged')
            print('Done.')
    def __str__(self):
        return ('MotifGopher instance with thresh={}, max_corr={}, '
                'confidence={}, r={}'.format(
                    self._thresh, self._max_corr, self._confidence, self._r
        ))

def cross_corr(arr1, arr2, padding):
    """Returns the max cross correlation of ARR1 and ARR2, with a
    displacement of magnitude in the interval [0, PADDING]."""
    return max(np.correlate(np.pad(arr1, padding), arr2))
def correlation(arr1, arr2, padding):
    """Returns the cosine of the angle between ARR1 and ARR2, after
    subjecting one of ARR1 or ARR2 to a displacement (shift) that
    maximizes its dot product with the other array.
    """
    return (
        cross_corr(arr1, arr2, padding)
        / ((
              np.sum(arr1)
            * np.sum(arr2)
        )**0.5)
    )