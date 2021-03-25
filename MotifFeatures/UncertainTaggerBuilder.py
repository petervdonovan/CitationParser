import numpy as np
import pandas as pd
import time
import random
import scipy
import itertools
import heapq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from MotifFeatures.MotifFeature import MotifFeature
from MotifFeatures.PositionFeature import PositionFeature

from MotifFeatures.Utils.algs import memo, set2vec, MeanAccumulator

from MotifFeatures.Utils.stats import t_ci, z_score

from numpy import inf

# Here, positive and negative infinity are mapped to negative and positive
# 999, respectively. This might seem backwards and counterintuitive, but
# the reason why this is done is that it is necessary to express that
# infinity is different from any finite number without actually having
# to represent infinity. If a series of numbers contains all negative
# values, including negative infinity, then replacing negative
# infinity with positive 999 will demonstrate that it is meaningfully
# different from any other value. I know this is a little inelegant, and
# I am open to suggestions for better workarounds.
INF_REPLACE = -999
NEG_INF_REPLACE = 999

# This is a small number. When a probability is lower than this, I
# assume impossibility. I know this is inelegant, but it saves time.
EPSILON = 1e-9

SECONDS_PER_MINUTE = 60

random.seed(18)

class UncertainTaggerBuilder:
    """Encapsulates the machinery required to
    * discover features from the countably infinite set of possible
      motif-based features, and
    * return a function that, when called on a string, returns
      a corresponding array of numbers in [0, 1]
    """
    def __init__(self,
                texts,
                tags,
                motifs,
                a=0.85,
                n=20,
                initial_max_separation=1,
                sort_sample_size=2000):
        """Initialize an UncertainTaggerBuilder with the required
        training data.
        TEXTS  - a sequence of strings
        TAGS   - a sequence of sequences of numbers in [0, 1]
                corresponding to TEXTS
        MOTIFS - a sequence of substrings used to create features
        A      - the first feature's probability of being chosen
        N      - the expected value of the size of a set of features,
                if approximating the set of possible features as infinite
        INITIAL_MAX_SEPARATION - the initial number of instances of a
                motif that can be detected ahead of or behind the
                current string position
        """
        self._texts = list(texts)
        self._tags  = list(tags)
        self._motifs = list(motifs)
        # This is the best OOB score yet observed.
        self._max_oob = -float('inf')
        self._a = a
        self._n = n
        self._guaranteed_n = 0
        self._features = [PositionFeature(True), PositionFeature(False)] + [
            MotifFeature(motif, i)
            for motif in self._motifs
            for i in range(-initial_max_separation, initial_max_separation)
        ]
        # Past states are stored for future recovery and for prediction
        # of the performance of future states. States are mapped to
        # performances.
        self._states = dict()

        self._model = RandomForestRegressor(
            random_state=random.randint(0, 1000), verbose=1, n_jobs=-1,
            max_features='sqrt',
            oob_score=True
        )
        self._importance_sort(sort_sample_size)
        # Key invariant: The features with the highest rankings are at
        # the beginning of the list.
        self._rankings = {
            f : -i for i, f in enumerate(self._features)
        }
        self.scores = list()
        # Invariant: self._best_feature_set must be in descending order
        # by importance.
        self._best_feature_set = self._features
    def _importance_sort(self, sort_sample_size):
        """Sort the features stored in SELF by their importances."""
        quickmodel = RandomForestRegressor(
            random_state=random.randint(0, 1000), verbose=1, n_jobs=-1,
            max_features='sqrt', max_samples=0.2, min_samples_split=8,
            oob_score=True)
        sample_idx = random.sample(
            list(range(len(self._texts))),
            k=min(sort_sample_size, len(self._texts))
        )
        sample_texts = [self._texts[i] for i in sample_idx]
        sample_y = get_y([self._tags[i] for i in sample_idx])
        quickmodel.fit(
            get_X(sample_texts, self._features),
            sample_y
        )
        self._max_oob = max(self._max_oob, quickmodel.oob_score_)
        self._features = sort_a_by_b(
            self._features, quickmodel.feature_importances_)

    def _sort(self):
        """Sort the features in descending order by their rankings."""
        self._features.sort(
            reverse=True,
            key=lambda f: self._rankings[f]
        )
    def _get_r(self):
        # This is just the geometric sum formula with the linearity of
        # expectation
        return 1 - self._a / (self._n - self._guaranteed_n)
    def _random_candidate_feature_set(self):
        """Returns a random feature set of nonzero length.
        """
        ret = self._best_feature_set[:self._guaranteed_n]
        p = self._a
        r = self._get_r()
        i = 0
        while i < len(self._features):
            if self._features[i] not in ret:
                if random.random() < p:
                    ret.append(self._features[i])
                p *= r
            i += 1
        if len(ret) == 0:
            return self._random_candidate_feature_set()
        return frozenset(ret)
    def _new_max_oob(self, ordered_feature_set):
        """Carries out the operations that correspond to discovering a new
        high-performing feature set. (This is a private helper function
        to IMPROVE.)
        ORDERED_FEATURE_SET must be in the same order as was used to
        most recently train the model.
        """
        self._max_oob = self._states[frozenset(ordered_feature_set)].oob_score
        # Update the ideal length to be that of the high-performing feature
        # set.
        self._n = len(ordered_feature_set)
        # Update the ordered list of best features.
        feature_importances = self._model.feature_importances_
        assert len(feature_importances) == len(ordered_feature_set)
        self._best_feature_set = ordered_feature_set
        self._best_feature_set = sort_a_by_b(
            self._best_feature_set, feature_importances
        )
        print('DEBUG: Max OOB score updated to {}'.format(
            self._max_oob))
        # Update the current list of features with their successors.
        for feature in ordered_feature_set:
            successor = None
            try:
                successor = feature.successor()
            except AttributeError:
                pass # This feature does not support successors.
            if successor and successor not in self._features:
                self._features.append(successor)
                self._rankings[successor] = self._rankings[feature] - 1
        self._sort()
    def _update_rankings(self, feature_set, oob_score):
        """Update the rankings for the features in FEATURE_SET according
        to whether the OOB score associated with them is good.
        """
        for f in feature_set:
            if f not in self._best_feature_set[:self._guaranteed_n]:
                z = z_score(
                    self.scores,
                    oob_score
                )
                if np.isfinite(z):
                    self._rankings[f] += z
        self._sort()
    def _improve(self):
        """Tests a new subset of the possible features.
        """
        y = get_y(self._tags)
        feature_set = self._random_candidate_feature_set()
        ordered_feature_set = list(feature_set)
        X = get_X(self._texts, ordered_feature_set, cache='self._texts')
        if feature_set not in self._states:
            self._states[feature_set] = UTBStatePerformance(
                self._model.get_params())
            self._model.fit(X, y)
            oob = self._model.oob_score_
            self._states[feature_set].oob_score = oob
            if oob >= self._max_oob:
                self._new_max_oob(feature_set)
            self._update_rankings(feature_set, oob)
            self.scores.append(oob)
    def run(self, duration):
        """Improves the feature selection for DURATION minutes."""
        t0 = time.time()
        duration_s = duration * SECONDS_PER_MINUTE
        while self._guaranteed_n < self._n:
            self._improve()
            self._guaranteed_n = int(
                self._n * (time.time() - t0) / duration_s)
            print('N = {}. Guaranteed N = {}.\nBest OOB = {}.'.format(
                self._n,
                self._guaranteed_n,
                self._max_oob
            ))


    def _CV(self, features, k=5, confidence=0.95):
        """Return a confidence interval for an f-score for a K-fold
        CV.
        """
        f_scores = list()
        for a in range(k):
            self._model.fit(
                get_X(
                    [
                        self._texts[i] for i in range(len(self._texts))
                        if i % k == a],
                    features,
                    cache='train partition {}/{}'.format(a, k)
                ),
                get_y(
                    [self._tags[i] for i in range(len(self._texts))
                        if i % k == a]
                )
            )
            actual = get_y(
                [self._tags[i] for i in range(len(self._texts))
                    if i % k != a]
            )
            pred = self._model.predict(get_X(
                [self._texts[i] for i in range(len(self._texts))
                    if i % k != a],
                features,
                cache='pred partition {}/{}'.format(a, k)
            ))
            print('DEBUG: pred: ', pred[:20])
            print('DEBUG: actual: ', actual[:20])
            pred = [round(p) for p in pred]
            print('DEBUG: precision: ', sum(pred[i] and actual[i]
                for i in range(len(actual))) / max(sum(pred), 0.00001))
            print('DEBUG: recall: ', sum(actual[i] and pred[i]
                for i in range(len(actual))) / max(sum(actual), 0.00001))
            f_scores.append(f1_score(actual, pred))
        return t_ci(f_scores, 1 - confidence)

class UTBStatePerformance:
    """Captures the performance of a set of features with a given set of
    random forest parameters.
    """
    def __init__(self, rf_params, oob_score=None, cv_score=None):
        if cv_score:
            assert len(self.cv_score) == 2, \
                'CV score must be a confidence interval.'
        self.rf_params = rf_params
        self.oob_score = oob_score
        self.cv_score = cv_score
    def mean_cv(self):
        """Returns a best estimate of the CV score associated with SELF.
        """
        if self.cv_score:
            assert len(self.cv_score) == 2, \
                'CV score must be a confidence interval.'
            return (self.cv_score[0] + self.cv_score[1]) / 2
        return None

def get_y(tags):
    """Returns the labels as an ndarray."""
    return np.concatenate(list(tags))
def get_X(texts, features, cache=None):
    """Returns a design matrix."""
    def rm_inf(arr):
        arr[arr==-inf] = NEG_INF_REPLACE
        arr[arr==inf]  = INF_REPLACE
        return arr
    return np.concatenate([
        rm_inf(feature.featurize_all(texts, cache=cache))[:,np.newaxis]
        for feature in features
    ], axis=1)
def sort_a_by_b(a, b):
    """Return the A sorted in descending order according to
    the values in B. (The ith element of B must be the
    priority level of the ith element of A.)
    """
    priorities = dict()
    for i, element in enumerate(a):
        priorities[element] = b[i]
    return sorted(
        a,
        reverse=True,
        key=lambda f: priorities[f]
    )