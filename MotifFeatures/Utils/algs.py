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

def rm_inf(seq):
    """Returns SEQ with infinite values replaced with large or small
    values.
    """
    try:
        minimum = min(val for val in seq if val != -float('inf'))
        maximum = max(val for val in seq if val != float('inf'))
    except ValueError:
        return np.array([0 for _ in range(len(seq))])
    for i, val in enumerate(seq):
        if val < minimum:
            seq[i] = minimum - 1
        elif val > maximum:
            seq[i] = maximum + 1
    return seq

def set2vec(sequence, subset):
    """Returns the logical (0/1) vector representation of a subset of
    SEQUENCE.
    """
    ret = np.zeros(len(sequence))
    for i, item in enumerate(sequence):
        if item in subset:
            ret[i] = 1
    return ret