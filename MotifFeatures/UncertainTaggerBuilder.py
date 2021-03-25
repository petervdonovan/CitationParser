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
                step=5,
                random_step=20,
                random_step_n=20,
                knn=10,
                knn_z=3,
                initial_max_separation=1,
                sort_sample_size=2000):
        """Initialize an UncertainTaggerBuilder with the required
        training data.
        TEXTS  - a sequence of strings
        TAGS   - a sequence of sequences of numbers in [0, 1]
                corresponding to TEXTS
        MOTIFS - a sequence of substrings used to create features
        STEP   - the step by which the maximum possible number of
                features increases with each iteration
        KNN    - the number of nearest neighbors to use when determinnig
                whether a given set of features is likely to have the
                maximum performance observed so far or better
        KNN_Z  - the minimum z-score of the maximum performance relative
                to the K sampled feature sets required to assume that
                no feature set similar to the K sampled feature sets
                can exceed the current maximum performance
        INITIAL_MAX_SEPARATION - the initial number of instances of a
                motif that can be detected ahead of or behind the
                current string position
        """
        self._texts = list(texts)
        self._tags  = list(tags)
        self._motifs = list(motifs)
        self._max_size = step
        self._step = step
        self._random_step = random_step
        self._random_step_n = random_step_n
        self._knn = knn
        self._knn_z = knn_z
        # This is the best OOB score yet observed.
        self._max_oob = -float('inf')
        self._features = [
            MotifFeature(motif, i)
            for motif in self._motifs
            for i in range(-initial_max_separation, initial_max_separation)
        ] + [PositionFeature(True), PositionFeature(False)]
        self._model = RandomForestRegressor(
            random_state=random.randint(0, 1000), verbose=1, n_jobs=-1,
            max_features='sqrt',
            oob_score=True)
        quickmodel = RandomForestRegressor(
            random_state=random.randint(0, 1000), verbose=1, n_jobs=-1,
            max_features='sqrt', max_samples=0.2, min_samples_split=8,
            oob_score=True
        )
        # Past states are stored for future recovery and for prediction
        # of the performance of future states. States are mapped to
        # performances.
        self._states = dict()

        # This is the only part of the init method that is more than
        # just boilerplate. This section of the method sorts the features
        # by an estimate of their importance, and it is crucial for
        # helping the program converge to good answers quickly.
        self._importances = dict()
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
            self._importances[feature] = \
                MeanAccumulator(quickmodel.feature_importances_[i])
        print('DEBUG: Time to get feature importances: {} seconds.'.format(
            time.time() - t0
        ))
        self._features.sort(
            reverse=True,
            key=lambda f: self._importances[f]
        )
    def _candidate_feature_sets(self):
        """Returns a list of all feature sets that are currently
        candidates for testing. (This is a private helper function to
        IMPROVE.)
        """
        return [
            frozenset(feature_tuple)
            for n_features in range(1, self._max_size + 1)
            for feature_tuple in itertools.combinations(
                self._features[:self._max_size], n_features
            )
            if ( # If it did not even receive a score, a feature set should be
                 # reconsidered.
                frozenset(feature_tuple) not in self._states
                or self._states[frozenset(feature_tuple)].oob_score is None
            )
        ]
    def _random_candidate_feature_sets(self):
        """Returns a list of RANDOM_STEP_N feature sets of size up to
        the current max size plus RANDOM_STEP. (This is a private helper
        function to IMPROVE.)
        """
        return [
            frozenset([
                feature for feature in
                self._features[:self._max_size + self._random_step]
                if random.random() > 0.5
            ])
            for _ in range(self._random_step_n)
        ]
    def _new_max_oob(self, feature_set):
        """Carry out the operations that correspond to discovering a new
        high-performing feature set. (This is a private helper function
        to IMPROVE.)
        """
        self._max_oob = self._states[feature_set].oob_score
        self._states[feature_set].cv_score = self._CV(feature_set)
        print('DEBUG: Max OOB score updated to {}'.format(
            self._max_oob))
        for feature in feature_set:
            successor = None
            try:
                successor = feature.successor()
            except AttributeError:
                pass # This feature does not support successors.
            if successor and successor not in self._features:
                self._features.insert(
                    self._max_size, successor)
                self._importances[successor] = MeanAccumulator(0)
    def improve(self):
        """Tests a new group of subsets of the set of all possible
        features.
        """
        y = get_y(self._tags)
        for feature_set in (self._random_candidate_feature_sets()
                          + self._candidate_feature_sets()       ):
            ordered_feature_set = list(feature_set)
            X = get_X(self._texts, ordered_feature_set, cache='self._texts')
            if feature_set not in self._states:
                self._states[feature_set] = UTBStatePerformance(
                    self._model.get_params())
            if (    (
                        feature_set not in self._states
                        # The following line allows states that have been
                        # deemed not promising to be reassessed.
                        or not self._states[feature_set].oob_score
                    )
                    and self._is_promising(feature_set)
                    ):
                self._model.fit(X, y)
                for idx, importance in enumerate(
                        self._model.feature_importances_):
                    self._importances[ordered_feature_set[idx]].add(importance)
                self._states[feature_set].oob_score = self._model.oob_score_
                if self._states[feature_set].oob_score >= self._max_oob:
                    self._new_max_oob(feature_set)
            else:
                print('DEBUG: A feature set did not seem promising.')
        self._max_size += self._step
    def _k_most_similar(self, k, feature_set):
        """Returns the K most similar feature sets to FEATURE_SET that
        have OOB scores (or all of them if there are fewer of them than
        K).
        """
        # This list "is" a min heap according to the heap ADT defined by
        # the heapq module.
        pq = []
        for other in self._states:
            if self._states[other].oob_score:
                # The most distant features will rise to the top because
                # this is a min heap. (Note the minus sign.)
                heapq.heappush(
                    pq,
                    (-self._distance(feature_set, other), other)
                )
                # The distant features at the top of the heap are then
                # removed.
                if len(pq) > k:
                    heapq.heappop(pq)
        return [item[1] for item in pq]
    def _distance(self, set1, set2):
        """Return the distance between SET1 and SET2, where both sets
        are subsets of the set of the first MAX_SIZE + RANDOM_STEP
        elements.
        """
        poss_features = self._features[:self._max_size + self._random_step]
        return scipy.spatial.distance.cosine(
            set2vec(poss_features, set1,
                weights=self._importances),
            set2vec(poss_features, set2,
                weights=self._importances)
        )
    def _is_promising(self, feature_set):
        """Returns True iff there is not enough information available to
        determine that FEATURE_SET does not contain the right features
        to outperform all other feature sets.
        """
        similar = self._k_most_similar(self._knn, feature_set)
        if len(similar) < self._knn:
            # There is insufficient information to reject FEATURE_SET
            return True
        scores = [self._states[fset].oob_score for fset in similar]
        return z_score(scores, self._max_oob) <= self._knn_z

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