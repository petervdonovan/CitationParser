import datetime
import random
import os.path
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
from scipy import stats

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
            max_corr=0.8,
            confidence=0.99,
            r=0.75,
            saveto='./'):
        """Initializes the MotifGopher with the variables that determine
        its behavior. These parameters ought not to change for a given
        MotifGopher instance.
        TEXTS - The texts to be searched for motifs.
        THRESH - The minimum frequency of a motif for it to be
                remembered by the MotifGopher. Frequency is calculated
                on a per-text basis, meaning that two appearances in a
                single text do not count for anything more than just one
                appearance.
        MAX_CORR - The maximum allowable cosine similarity between two
                included motifs. Calculated with shifts, in much the
                same fashion as cross-correlation, with the vectors
                being binary sequences of match-or-no-match with the
                motif of interest.
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
        self._max_corr = max_corr
        self._confidence = confidence
        self._r = r
        self._saveto = saveto
        self.found_motifs = dict()
        self._name = datetime.datetime.now().strftime(
            'Gopher_%I-%M%p_%d_%b_%Y'
        )
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
    def _get_motif_vecs(self):
        """Returns a dictionary of logical ndarrays representing the
        indices of matches in the concatenated group of all texts.

        Note: Uses a lot of space. On the order of millions of integers.
        """
        concatenated = ''.join(self._texts)
        ret = {
            motif: np.zeros(len(concatenated))
            for motif in self.found_motifs
        }
        for key in ret:
            idx = concatenated.find(key)
            while idx != -1:
                ret[key][idx] = 1
                idx = concatenated.find(key, idx + 1)
        return ret
    def purge(self, verbose=False):
        """Purges the found motifs of excessively correlated pairs;
        includes the longer motif whenever two motifs are excessively
        correlated. It is recommended to call this method only once,
        after collecting a satisfactory number of motifs.

        WARNING: The implementation I have right now is what might be
        called the "naive implementation." Yes, I am naive: I'm
        nineteen. So, it's kind of slow, and it takes quadratic time wrt
        the number of found motifs.
        """
        corr_padding = max(len(motif) for motif in self.found_motifs)
        motif_vecs = self._get_motif_vecs()
        keys = list(self.found_motifs.keys())
        keys.sort(key=lambda s: len(s), reverse=True)
        # Here, while loops are necessary (as opposed to for loops)
        # because the length of the list changes as items are removed.
        i = 0
        while i < len(keys) - 1:
            j = i + 1
            while j < len(keys):
                assert len(keys[i]) >= len(keys[j])
                corr = correlation(
                    motif_vecs[keys[i]],
                    motif_vecs[keys[j]],
                    corr_padding
                )
                if corr > self._max_corr:
                    if verbose:
                        print('Deleting "{}" because it was too similar to '
                              '"{}" (similarity: {:.4f})'
                            .format(keys[j], keys[i], corr))
                        if i % 10 == 0:
                            print('Searched {}% of the keys...'.format(
                                100 * i / len(keys)
                            ))
                    del self.found_motifs[keys[j]]
                    keys.remove(keys[j])
                j += 1
            i += 1

    def log(self, message):
        """Appends MESSAGE to the file .gopherlogs, prefixed with the
        gopher's name.
        """
        with open(os.path.join(self._saveto, '.gopherlogs'), 'w') as f:
            f.write('{}: {}\n'.format(self._name, message))
    def save(self):
        """Saves a dictionary of found motifs and their frequencies to a
        pickle file.
        """
        path = os.path.join(self._saveto, self._name + '_motifs.pickle')
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
            if 'y' in input('Would you like to save and exit? ').lower():
                break
        self.save()
def correlation(arr1, arr2, padding):
    """Returns the cosine of the angle between ARR1 and ARR2, after
    subjecting one of ARR1 or ARR2 to a displacement (shift) that
    maximizes its dot product with the other array.
    """
    return (
        max(np.correlate(np.pad(arr1, padding), arr2))
        / ((
              np.sum(arr1)
            * np.sum(arr2)
        )**0.5)
    )