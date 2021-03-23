import numpy as np
import pandas as pd
import time
import random
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

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
    def __init__(self, texts, tags, motifs, seed=0, n_for_sort=1000):
        """Initialize an UncertainTaggerBuilder with the required
        training data.
        TEXTS  - a sequence of strings
        TAGS   - a sequence of sequences of numbers in [0, 1]
                 corresponding to TEXTS
        MOTIFS - a sequence of substrings used to create features
        """
        self._texts = list(texts)
        self._tags  = list(tags)
        self._motifs = list(motifs)
        self._motifs.sort(
            reverse=True,
            key=lambda motif: sum(
                1 if motif in text else 0 for text in self._texts
        ))
        self._features = [
            MotifFeature(motif, i)
            for motif in self._motifs
            for i in range(-1, 1)
        ]
        self._model = RandomForestRegressor(
            random_state=random.randrange(0, 1000), verbose=1, n_jobs=-1,
            max_features='log2')
        # States maps sets of features to 90% confidence intervals for
        # their performances.
        self._states = dict()

        importances = dict()
        t0 = time.time()
        m = 7 # Any prime number should be fine.
        for a in range(m):
            print('{}/{} of the way finished sorting features after {} '
                  'seconds'.format(a, m, time.time() - t0))
            selected_features = [
                self._features[i] for i in range(len(self._features))
                if i % m == a
            ]
            idx = random.sample(
                [i for i in range(len(self._texts))],
                k=n_for_sort)
            tags  = [self._tags[i]  for i in idx]
            texts = [self._texts[i] for i in idx]
            self._model.fit(
                self._X(texts, selected_features),
                self._y(tags)
            )
            for i, feature in enumerate(selected_features):
                importances[feature] = \
                    self._model.feature_importances_[i]
        self._features.sort(
            reverse=True,
            key=lambda f: importances[f]
        )

    def _y(self, tags):
        """Returns the labels as an ndarray."""
        return np.concatenate(list(tags))
    def _X(self, texts, features):
        """Returns a design matrix in the form of a DataFrame."""
        data={
            str(feature): rm_inf(feature.featurize_all(texts))
            for feature in features
        }
        data['pos'] = pd.Series(
            i
            for text in texts
            for i in range(len(text))
        )
        data['reverse-pos'] = pd.Series(
            i
            for text in texts
            for i in range(len(text)-1, -1, -1)
        )
        return pd.DataFrame(data=data)
    def _CV(self, features, k=5, confidence=0.9):
        """Return a confidence interval for an f-score for a K-fold
        CV.
        """
        f_scores = list()
        index = list(range(len(self._texts)))
        random.shuffle(index)
        for a in range(k):
            validation_idx = [
                index[i] for i in range(len(index)) if i % k == a
            ]
            train_idx = [
                index[i] for i in range(len(index)) if i % k != a
            ]
            self._model.fit(
                self._X(
                    [self._texts[idx] for idx in train_idx],
                    features),
                self._y(
                    [self._tags[idx]  for idx in train_idx]
                )
            )
            actual = self._y(
                [self._tags[idx]  for idx in validation_idx]
            )
            pred = self._model.predict(self._X(
                [self._texts[idx] for idx in validation_idx],
                features
            ))
            print('DEBUG: pred: ', pred[:20])
            print('DEBUG: actual: ', actual[:20])
            pred = [round(p) for p in pred]
            print('DEBUG: precision: ', sum(pred[i] and actual[i] for i in range(len(actual))) / sum(pred))
            print('DEBUG: recall: ', sum(actual[i] and pred[i] for i in range(len(actual))) / sum(actual))
            f_scores.append(f1_score(actual, pred))
        return t_ci(f_scores, 1 - confidence)
        

def rm_inf(seq):
    """Returns a copy of SEQ with infinite values replaced with
    large or small values.
    """
    try:
        minimum = min(val for val in seq if val != -float('inf'))
        maximum = max(val for val in seq if val != float('inf'))
    except ValueError:
        return pd.Series(0 for _ in range(len(seq)))
    return pd.Series([
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