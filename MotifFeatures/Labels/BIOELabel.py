from gensim.utils import simple_preprocess
from Utils.algs import memo
import numpy as np

class BIOELabel:
    """Encapsulates logic for generating BIOES labels for a given text
    and sequence of named entities to search for.
    """
    def __init__(self, text, entities):
        """Initializes a BIOSLabel with a raw text TEXT and a sequence
        of named entities ENTITIES.
        """
        self._text = text
        self._entities = entities
    
    @memo
    def _intervals(self):
        """Returns an iterable of tuples: the beginnings and endings
        (start inclusive, stop not inclusive, as is the custom) of the
        non-null entity matches in TEXT.
        Throws an AssertionError in the (absolutely possible!) case that
        two intervals overlap. If this happens, it isn't necessarily an
        algorithmic issue; instead, it may be a problem with the dataset
        itself or our assumptions about how the dataset should be
        structured. Named entities shouldn't overlap with each other.
        """
        ret = [
            minimal_matching_substring(self._text, entity)
            for entity in self._entities
        ]
        ret = [
            ivl for ivl in ret if None not in ivl
        ]
        assert all(
            a == b or (
                (a[0] < a[1] <= b[0]) or (b[0] < b[1] <= a[0])
            )
            for a in ret
            for b in ret
        ), "Intervals must not overlap."
        return ret

    def _tags_helper(self, condition):
        """Returns a numpy logical array noting the locations that
        should be given a particular tag. Whether a location should be
        given a particular tag is determined by CONDITION, a function
        which takes a string index and a sequence of tuple intervals as
        input and outputs a boolean.
        """
        ret = np.zeros(len(self._text))
        for i in range(len(self._text)):
            if condition(i, self._intervals()):
                ret[i] = 1
        return ret
    def B_tags(self):
        """Returns a numpy logical (0 and 1) array noting the locations
        that should be tagged as B. Values in this array may be
        interpreted as degrees of certainty that the given characters
        correspond to beginnings of minimal matching substrings to named
        entities. Alternatively, they may simply be interpreted as
        one-hots.
        """
        return self._tags_helper(
            lambda i, intervals: any(
                ivl[0] == i
                for ivl in intervals
            )
        )
    def I_tags(self):
        """Returns a numpy logical (0 and 1) array noting the locations
        that should be tagged as I. Values in this array may be
        interpreted as degrees of certainty that the given characters
        correspond to insides of minimal matching substrings to named
        entities. Alternatively, they may simply be interpreted as
        one-hots.
        """
        return self._tags_helper(
            lambda i, intervals: any(
                ivl[0] < i < ivl[1]
                for ivl in intervals
            )
        )
    def O_tags(self):
        """Returns a numpy logical (0 and 1) array noting the locations
        that should be tagged as O. Values in this array may be
        interpreted as degrees of certainty that the given characters
        are not part of minimal matching substrings to named
        entities. Alternatively, they may simply be interpreted as
        one-hots.
        """
        return self._tags_helper(
            lambda i, intervals: all(
                not (ivl[0] <= i < ivl[1])
                for ivl in intervals
            )
        )
    def E_tags(self):
        """Returns a numpy logical (0 and 1) array noting the locations
        that should be tagged as E. Values in this array may be
        interpreted as degrees of certainty that the given characters
        correspond to beginnings of minimal matching substrings to named
        entities. Alternatively, they may simply be interpreted as
        one-hots.
        """
        return self._tags_helper(
            lambda i, intervals: any(
                ivl[1] == i
                for ivl in intervals
            )
        )

def contains(a, b):
    """Returns whether A contains B. (Code snippet adapted from
    https://stackoverflow.com/questions/3847386)
    """
    a, b = simple_preprocess(a), simple_preprocess(b)
    for start in range(len(a) - len(b) + 1):
        for element_idx in range(len(b)):
            if a[start+element_idx] != b[element_idx]:
                break
        else:
            return True
    return False

def minimal_matching_substring(text, s, preproc=simple_preprocess):
    """Returns the start and stop indices of the smallest substring of
    TEXT that matches S. PREPROC is applied to both the substring of
    TEXT and S to determine if they match. Returns None if there is no
    matching substring.
    text - The raw text to be searched
    s    - The string to be searched for
    preproc - A function that takes in a string and outputs a list of
              tokens.
    """
    def binary_search(at_least_condition, greater_condition, lo, hi):
        while lo <= hi - 1:
            mid = (lo + hi) // 2
            if at_least_condition(mid):
                if not greater_condition(mid):
                    return mid
                lo = mid + 1
            else:
                hi = mid
        return None
    if not contains(text, s):
        return None, None
    start = binary_search(
        lambda idx: contains(text[idx:len(text)], s),
        lambda idx: contains(text[idx+1:len(text)], s),
        lo=0,
        hi=len(text)
    )
    stop = binary_search(
        lambda idx: not contains(text[start:idx-1], s),
        lambda idx: not contains(text[start:idx], s),
        lo=start+1,
        hi=len(text)+1
    )
    return start, stop
