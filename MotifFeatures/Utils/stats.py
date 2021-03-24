import math
import numpy as np
from scipy import stats

def t_ci(seq, alpha):
    return stats.t.interval(
        alpha, len(seq) - 1, np.average(seq),
        stats.sem(seq)
    )

def z_score(seq, other):
    return (other - np.mean(seq)) / np.std(seq)

def fisher_r2z(r):
    """Transforms a correlation coefficient to z-space.
    TODO: Include further study and justification for this
    transformation.
    """
    return 0.5 * math.log((1 + r) / (1 - r))

def r_ci(r, n, confidence):
    """Returns an confidence interval with confidence CONFIDENCE for a
    correlation coefficient.
    """
    z = fisher_r2z(r)
    z_std_u = math.sqrt(1 / (n - 3))
    delta_z = stats.norm