import pandas as pd
import numpy as np

class MotifFeature:
    """Encapsulates the machinery required to take strings and give
    sequences of attributes of the string positions,
    where the attributes of a string position are determined by the
    motifs that surround it.
    """
    def __init__(self, motif, i):
        """Initializes a MotifFeature instance representing the feature
        that is the position of the ith instance of MOTIF relative to
        the position of interest (where i may be negative).
        """
        self._motif = motif
        self._i     = i
        self._cache = dict()
    def __hash__(self):
        return hash((self._i, self._motif))
    def __eq__(self, other):
        return self._i == other._i and self._motif == other._motif
    def featurize(self, s):
        """Returns an array of feature values corresponding to each
        position in S.
        """
        ret = np.zeros(len(s))
        current_deltapos = find_ith(self._motif, s, 0, self._i)
        ret[0] = current_deltapos
        for start in range(1, len(s)):
            current_deltapos -= 1
            if s.startswith(
                    self._motif,
                    start if self._i >= 0 else start - 1
                    ):
                current_deltapos = (
                    find_ith(self._motif, s, start, self._i)
                    - start)
            ret[start] = current_deltapos
        return ret
    def featurize_all(self, texts, cache=None):
        """Returns an array of the feature values corresponding to each
        position in each text in TEXTS.
        CACHE - the key that will be used to retrieve the same result
            from this Feature in the future
        """
        if cache and cache in self._cache:
            return self._cache[cache]
        ret = np.concatenate([self.featurize(text) for text in texts])
        if cache:
            self._cache[cache] = ret
        return ret
    def successor(self):
        """Returns a MotifFeature that is similar to SELF but not the
        same.
        """
        if self._i < 0:
            return MotifFeature(self._motif, self._i-1)
        return MotifFeature(self._motif, self._i+1)

    def __str__(self):
        return '{}th_"{}"'.format(self._i, self._motif)

def find_ith(search, s, start, i):
    """Find the position of the ith instance of SEARCH in S at or after
    START. Returns infinity if there is no such instance.

    Extends naturally for negative I, i.e., returns the position of
    abs(i)th instance strictly _before_ START for negative i. For I=0,
    behaves similarly as for the strictly positive integers.
    """
    if i < 0:
        pos_transform = lambda pos: len(s) - pos - len(search)
        return pos_transform(
                find_ith(
                    search[::-1],
                    s[::-1],
                    pos_transform(start - 1),
                    abs(i) - 1)
            )
    finder = find_after(search, s, start)
    ret = None
    idx = 0
    while idx <= i:
        try:
            ret = next(finder)
        except StopIteration:
            return float('inf')
        idx += 1
    return ret


def find_after(search, s, start):
    """Yield the positions of S that are after START where perfect
    matches for SEARCH can be found.
    """
    pos = s.find(search, start)
    while pos != -1:
        yield pos
        pos = s.find(search, pos + 1)