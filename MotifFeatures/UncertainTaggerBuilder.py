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
                memory=10,
                reset_n=5,
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
        self._memory = memory
        self._reset_n = reset_n
        self._resets = list()
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
        self._rankings = {
            f : -i for i, f in enumerate(self._features)
        }
        self.scores = list()
        # Key invariant: The features with the highest rankings are at
        # the beginning of the list.
    def _importance_sort(self, sort_sample_size):
        """Sort the features stored in SELF by their importances."""
        quickmodel = RandomForestRegressor(
            random_state=random.randint(0, 1000), verbose=1, n_jobs=-1,
            max_features='sqrt', max_samples=0.2, min_samples_split=8,
            oob_score=True)
        importances = dict()
        sample_idx = random.sample(
            list(range(len(self._texts))),
            k=min(sort_sample_size, len(self._texts))
        )
        sample_texts = [self._texts[i] for i in sample_idx]
        sample_y = get_y([self._tags[i] for i in sample_idx])
        t0 = time.time()
        quickmodel.fit(
            get_X(sample_texts, self._features),
            sample_y
        )
        print('DEBUG: Time to fit to features: {} seconds.'.format(
            time.time() - t0))
        t0 = time.time()
        for i, feature in enumerate(self._features):
            importances[feature] = \
                MeanAccumulator(quickmodel.feature_importances_[i])
        print('DEBUG: Time to get feature importances: {} seconds.'.format(
            time.time() - t0
        ))
        self._features.sort(
            reverse=True,
            key=lambda f: importances[f]
        )
    def _sort(self):
        """Sort the features in descending order by their rankings."""
        self._features.sort(
            reverse=True,
            key=lambda f: self._rankings[f]
        )
    def _get_r(self):
        # This is just the geometric sum formula with the linearity of
        # expectation
        return 1 - self._a / self._n
    def _improvement(self):
        """Return a number that is positive iff it seems like scores
        might be improving.
        """
        if len(self.scores) < self._memory:
            return 1
        return scipy.stats.pearsonr(
            range(self._memory), self.scores[-self._memory:]
        )[0]
    def _reset(self):
        """Reset internal parameters to increase the probability of
        replicating results that have been positive.
        """
        if len(self.scores) < self._reset_n:
            return
        min_required_score = sorted(self.scores)[-self._reset_n]
        sample = [
            feature_set for feature_set in self._states
            if self._states[feature_set].oob_score >= min_required_score
        ]
        self._resets.append(len(self.scores))
        # Make future feature sets have about the same length as the
        # high-performing ones
        self._n = np.mean([len(feature_set) for feature_set in sample])
        # Bring the features that commonly appear in high-performing feature
        # sets to the front of the feature list
        desirable_features = dict()
        for feature_set in sample:
            for feature in feature_set:
                desirable_features[feature] = \
                    desirable_features.get(feature, 0) + 1
        max_ranking = self._rankings[self._features[0]]
        for feature in desirable_features:
            self._rankings[feature] = max_ranking + desirable_features[feature]
        self._sort()
    def _random_candidate_feature_set(self):
        """Returns a feature set selected from the highest-ranked N
        features. Does not return a zero-length feature set.
        """
        ret = list()
        p = self._a
        r = self._get_r()
        i = 0
        while p >= EPSILON and i < len(self._features):
            if random.random() < p:
                ret.append(self._features[i])
            p *= r
            i += 1
        if len(ret) == 0:
            return self._random_candidate_feature_set()
        return frozenset(ret)
    def _new_max_oob(self, feature_set):
        """Carry out the operations that correspond to discovering a new
        high-performing feature set. (This is a private helper function
        to IMPROVE.)
        """
        self._max_oob = self._states[feature_set].oob_score
        print('DEBUG: Max OOB score updated to {}'.format(
            self._max_oob))
        for feature in feature_set:
            successor = None
            try:
                successor = feature.successor()
            except AttributeError:
                pass # This feature does not support successors.
            if successor and successor not in self._features:
                self._features.append(successor)
                self._rankings[successor] = self._rankings[feature]
        self._sort()
    def _update_rankings(self, feature_set, oob_score):
        """Update the rankings for the features in FEATURE_SET according
        to whether the OOB score associated with them is good.
        """
        for f in feature_set:
            z = z_score(
                self.scores[max(0, len(self.scores) - self._memory):],
                oob_score
            )
            if np.isfinite(z):
                self._rankings[f] += z
        self._sort()
    def improve(self):
        """Tests a new subset of the possible features.
        """
        if self._improvement() <= 0:
            print('RESETTING because scores are not improving.')
            self._reset()
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