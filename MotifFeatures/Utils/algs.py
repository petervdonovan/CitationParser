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