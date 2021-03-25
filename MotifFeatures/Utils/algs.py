import numpy as np

def memo(method):
    """Boilerplate for memoization. I could have used a library, but as
    a student I prefer to see what is going on.

    Most usages of this will be micro-optimizations by my standards --
    changes by a constant factor, not time complexity changes -- but
    this will prevent repetition.
    """
    def memoized(self, *args):
        try:
            results = self._memo_results
        except:
            results = dict()
            self._memo_results = results
        args = tuple(args)
        if args not in results:
            results[args] = method(self, *args)
        return results[args]
    return memoized

def replace(seq, map):
    """Replace elements of SEQ that match keys of MAP to the
    corresponding values and return the result."""

def set2vec(sequence, subset, weights=None):
    """Returns a vector representation of a subset of SEQUENCE."""
    ret = np.zeros(len(sequence))
    for i, item in enumerate(sequence):
        if item in subset:
            weight = 1
            if weights:
                try:
                    weight = weights[item].mean()
                except AttributeError:
                    weight = weights[item]
            ret[i] = weight
    return ret

class MeanAccumulator:
    def __init__(self, initial=0):
        self.sum = initial
        self.n = 1
    def add(self, value):
        self.sum += value
        self.n += 1
    def mean(self):
        return self.sum / self.n
    def __lt__(self, other):
        return self.mean() < other.mean()
    def __gt__(self, other):
        return self.mean() > other.mean()
    def __eq__(self, other):
        return self.mean() == other.mean()