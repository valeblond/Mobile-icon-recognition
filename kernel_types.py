from kernel_class import Kernel
import numpy as np

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

class Exponential(Kernel):
    """
    Exponential kernel,
        K(x, y) = e^(-||x - y||/(s^2))
    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-np.sqrt(dists_sq) / self._sigma)

    def dim(self):
        return np.inf


class Quadratic(Kernel):
    """
    Rational quadratic kernel,
        K(x, y) = - sqrt(||x-y||^2 + c^2)
    where:
        c > 0
    """

    def __init__(self, c=1):
        self._c = c

    def _compute(self, data_1, data_2):
        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return (- np.sqrt(dists_sq + self._c))

    def dim(self):
        return None  # unknown?
