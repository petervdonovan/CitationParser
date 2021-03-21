import pandas as pd

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
    def featurize(self, s):
        """Return a Series of feature values corresponding to the string
        S.
        """
        return pd.Series(
            find_ith(self._motif, s, start, self._i) - start
            for start in range(len(s))
        )

def find_ith(search, s, start, i):
    """Find the position of the ith instance of SEARCH in S at or after
    START. Returns infinity if there is no such instance.

    Extends naturally for negative I, i.e., returns the position of
    abs(i)th instance strictly _before_ START for negative i. For I=0,
    behaves similarly as for the strictly positive integers.
    """
    if i < 0:
        pos_transform = lambda pos: len(s) - 1 - pos
        return pos_transform(
                find_ith(
                    search[::-1],
                    s[::-1],
                    pos_transform(start) + 1,
                    abs(i) - 1)
            ) - (len(search) - 1)
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