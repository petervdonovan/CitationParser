import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from MotifFeatures.MotifFeature import MotifFeature
from MotifFeatures.Utils.algs import memo

class UncertainTaggerBuilder:
    """Encapsulates the machinery required to
    * discover features from the countably infinite set of possible
      motif-based features, and
    * return a model (a function that, when called on a string, returns
      a corresponding array of numbers in [0, 1])
    """
    def __init__(self, texts, tags, motifs):
        """Initialize an UncertainTaggerBuilder with the required
        training data.
        TEXTS  - a sequence of strings
        TAGS   - a sequence of sequences of numbers in [0, 1]
                 corresponding to TEXTS
        MOTIFS - a sequence of substrings used to create features
        """
        self._texts = texts
        self._tags  = tags
        self._motifs = motifs
        self._features = [
            MotifFeature(motif, i)
            for i in range(-2, 2)
            for motif in motifs
        ]
    @memo
    def _labels(self):
        """Returns the labels as a Series object."""
        return pd.Series(np.concatenate(self._tags))
    def _X(self):
        """Returns a design matrix in the form of a DataFrame."""
        return pd.DataFrame(data={
            str(feature): pd.concat(
                feature.featurize(text)
                for text in self._texts
            )
            for feature in self._features
        })