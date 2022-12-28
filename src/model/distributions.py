from abc import ABC, abstractmethod

import numpy as np
import scipy


class Distribution(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self, *args, **kwargs) -> float:
        pass


class PoissonDistribution(Distribution):
    """Poisson distribution with fixed mean."""

    def __init__(self, lam: float = 100):
        """
        Construct a Poisson distribution object.
        :param lam: Mean of the distribution.
        """
        self.mean = lam

    def generate(self):
        return np.random.poisson(self.mean)


class NormalDistribution(Distribution):
    """Normal distribution with fixed mean and std."""

    def __init__(self, mu: float, sigma: float):
        """
        Construct a Normal distribution object.
        :param mu: Mean of the distribution.
        :param sigma: Std of the distribution.
        :return: An interger generated by the distribution
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        return np.ceil(np.random.normal(self.mu, self.sigma))


class GammaDistribution(Distribution):
    """Implementation of Gamma distribution"""

    def __init__(self, a, loc, scale):
        self.a = a
        self.loc = loc
        self.scale = scale

    def generate(self):
        return max(np.ceil(scipy.stats.gamma.rvs(self.a, self.loc, self.scale)), 0)


class NormDistribution(Distribution):
    """Implementation of Gamma distribution"""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def generate(self):
        return max(np.ceil(scipy.stats.norm.rvs(self.loc, self.scale)), 0)


class UniformDistribution(Distribution):
    """Implementation of Gamma distribution"""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def generate(self):
        return max(np.ceil(scipy.stats.uniform.rvs(self.start, self.end)), 0)


class BetaDistribution(Distribution):
    """Implementation of Gamma distribution"""

    def __init__(self, a, b, loc, scale):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def generate(self):
        return max(
            np.ceil(scipy.stats.beta.rvs(self.a, self.b, self.loc, self.scale)), 0
        )
