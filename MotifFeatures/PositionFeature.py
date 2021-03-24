import numpy as np

class PositionFeature:
    """This class describes an animal that walks and quacks like a
    MotifFeature but that instead measures distance from the beginning
    or end of a text.
    """
    def __init__(self, forward):
        """Initializes a PositionFeature instance representing the
        distance either forward or backward to an endpoint of a text.
        """
        self.forward = forward
    def __hash__(self):
        return hash(('PositionFeature instance', self.forward))
    def __eq__(self, other):
        return hash(self) == hash(other)
    def featurize(self, s):
        """Returns an array of feature values corresponding to each
        position in S.
        """
    def featurize_all(self, texts, cache=None):
        return np.array([
            i
            for text in texts
            for i in (
                range(len(text)) if self.forward
                else range(len(text)-1, -1, -1)
            )
        ])
    def __str__(self):
        return '{}PositionFeature'.format(
            'FORWARD' if self.forward else 'REVERSE')
