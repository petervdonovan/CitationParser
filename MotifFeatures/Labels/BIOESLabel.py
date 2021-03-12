from gensim.utils import simple_preprocess

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
