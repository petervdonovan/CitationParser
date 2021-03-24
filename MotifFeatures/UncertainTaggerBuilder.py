import numpy as np
import pandas as pd
import time
import random
import scipy
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from MotifFeatures.MotifFeature import MotifFeature
from MotifFeatures.Utils.algs import memo

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
                knn=10,
                knn_z=3,
                n_partitions=13,
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
        N_PARTITIONS - the number of partitions into which to divide
                the features when initially ranking them; preferably
                either 1 (if the dataset is small) or a prime number
                that is not too small and not too large
        INITIAL_MAX_SEPARATION - the initial number of instances of a
                motif that can be detected ahead of or behind the
                current string position
        """
        self._texts = list(texts)
        self._tags  = list(tags)
        self._motifs = list(motifs)
        self._max_size = step
        self._step = step
        self._motifs.sort(
            reverse=True,
            key=lambda motif: sum(
                1 if motif in text else 0 for text in self._texts
        ))
        self._features = [
            MotifFeature(motif, i)
            for motif in self._motifs
            for i in range(-initial_max_separation, initial_max_separation)
        ]
        self._model = RandomForestRegressor(
            random_state=random.randint(0, 1000), verbose=1, n_jobs=-1,
            max_features='sqrt')
        self._quickmodel = RandomForestRegressor(
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
        importances = dict()
        t0 = time.time()
        sample_idx = list(range(len(self._texts)))
        random.shuffle(sample_idx)
        n = min(sort_sample_size, len(self._texts))
        sample_texts = [self._texts[i] for i in sample_idx[:n]]
        sample_y = self._y([self._tags[i] for i in sample_idx[:n]])
        # It is advisable to reshuffle so that the CV partitions that contain
        # the texts sampled here do not have an advantage over the others.
        random.shuffle(sample_idx)
        self._texts = [self._texts[i] for i in sample_idx]
        self._tags = [self._tags[i] for i in sample_idx]
        for a in range(n_partitions):
            print('{}/{} of the way finished sorting features after {} '
                  'seconds'.format(a, n_partitions, time.time() - t0))
            # Because the features have already been sorted by how
            # common they are, each partition of the features should
            # be a representative cross section of them. It is also
            # crucial that 2*INITIAL_MAX_SEPARATION is not a factor of
            # N_PARTITIONS -- this is one reason why it is best for
            # N_PARTITIONS to be prime.
            selected_features = [
                self._features[i] for i in range(len(self._features))
                if i % n_partitions == a
            ]
            self._quickmodel.fit(
                self._X(sample_texts, selected_features),
                sample_y
            )
            for i, feature in enumerate(selected_features):
                importances[feature] = \
                    self._quickmodel.feature_importances_[i]
        self._features.sort(
            reverse=True,
            key=lambda f: importances[f]
        )
    def improve(self):
        """Tests a new group of subsets of the set of all possible
        features.
        """
        featuresets = [
            frozenset(feature_tuple)
            for n_features in range(1, self._max_size + 1)
            for feature_tuple in itertools.combinations(
                self._features[:self._max_size], n_features
            )
            if frozenset(feature_tuple) not in self._states
        ]
        Xs = [
            self._X(self._texts, featureset, cache='self._texts')
            for featureset in featuresets
        ]
        y = self._y(self._tags)
        oobs = list()
        for featureset, X in zip(featuresets, Xs):
            self._states[featureset] = UTBStatePerformance(
                self._model.get_params())
            self._quickmodel.fit(X, y)
            self._states[featureset].oob_score = self._quickmodel.oob_score_
            oobs.append(self._quickmodel.oob_score_)
        # TODO: Avoid looking at all featuresets >>>>>>>>>>
        cvs = list()
        for featureset in featuresets:
            self._states[featureset].cv_score = self._CV(featureset)
            cvs.append(self._states[featureset].mean_cv())
        plt.title('CV Scores vs. OOB Scores')
        plt.xlabel('OOB Score (R^2)')
        plt.ylabel('CV Score (F1)')
        plt.scatter(oobs, cvs)
        plt.show()
        # <<<<<<<<<<<<<<<<< END TODO <<<<<<<<<<<<<<<<
        plt.title('OOB Score Distribution for {} Featuresets'.format(
            len(featuresets)
        ))
        plt.xlabel('OOB Score (R^2)')
        plt.ylabel('Frequency')
        plt.hist(oobs)
        plt.show()
        self._max_size += self._step
    def _most_similar_feature_sets(self, k, feature_set):
        """Returns the K most similar feature sets to FEATURE_SET that
        have CV scores.
        """
        pq = []
        poss_features = list(set(
            feature for feature in fset for fset in self._states
        ))
        for fset in self._states:
            if self._states[fset].cv_score:
                heappush((scipy.spatial.distance.cosine(
                    set2vec()
                ), fset))

    def _y(self, tags):
        """Returns the labels as an ndarray."""
        return np.concatenate(list(tags))
    def _X(self, texts, features, cache=None):
        """Returns a design matrix."""
        data=[
            list(rm_inf(feature.featurize_all(texts, cache=cache)))
            for feature in features
        ]
        data.append([
            i
            for text in texts
            for i in range(len(text))
        ])
        data.append([
            i
            for text in texts
            for i in range(len(text)-1, -1, -1)
        ])
        return np.array(data).T
    def _CV(self, features, k=5, confidence=0.95):
        """Return a confidence interval for an f-score for a K-fold
        CV.
        """
        f_scores = list()
        for a in range(k):
            self._model.fit(
                self._X(
                    [
                        self._texts[i] for i in range(len(self._texts))
                        if i % k == a],
                    features,
                    cache='train partition {}/{}'.format(a, k)
                ),
                self._y(
                    [self._tags[i] for i in range(len(self._texts))
                        if i % k == a]
                )
            )
            actual = self._y(
                [self._tags[i] for i in range(len(self._texts))
                    if i % k != a]
            )
            pred = self._model.predict(self._X(
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


def rm_inf(seq):
    """Returns a copy of SEQ with infinite values replaced with
    large or small values.
    """
    try:
        minimum = min(val for val in seq if val != -float('inf'))
        maximum = max(val for val in seq if val != float('inf'))
    except ValueError:
        return np.array([0 for _ in range(len(seq))])
    return np.array([
        (
            minimum - 1 if val < minimum
            else (maximum + 1 if val > maximum else val)
        ) for val in seq
    ])

def t_ci(seq, alpha):
    return scipy.stats.t.interval(
        alpha, len(seq) - 1, np.average(seq),
        scipy.stats.sem(seq)
    )

def set2vec(sequence, subset):
    """Returns the logical (0/1) vector representation of a subset of
    SEQUENCE.
    """
    ret = np.zeros(len(set))
    for i, item in enumerate(sequence):
        if item in subset:
            ret[i] = 1
    return ret