"""Chapter 3, stochastic systems."""


import random
from collections.abc import Generator
from typing import NoReturn


class RandomProcess(random.Random):
    """Random process base class."""


def arma(p: int, q: int, N: int, process: None | Generator = None) -> NoReturn:
    """Auto-regressive Moving-Average."""
    raise NotImplementedError()


def ar(p: int, N: int, process: None | Generator = None) -> NoReturn:
    """Auto-regressive."""
    raise NotImplementedError()


def ma(q: int, N: int, process: None | Generator = None) -> NoReturn:
    """Moving Average Random (stochastic) process."""
    raise NotImplementedError()


def harmonic(A: int, process: None | Generator = None) -> NoReturn:
    """The harmonic random process."""
    raise NotImplementedError()


def white_noise(variance: float) -> NoReturn:
    """The harmonic random process.

    Page 93.

    """
    raise NotImplementedError()


def white_gaussian_noise() -> NoReturn:
    """A random process, a sequence of uncorrelated real-valued Gaussian random.

    Page 94.

    """
    raise NotImplementedError()


def bernoulli() -> NoReturn:
    """The Bernoulli process consists of a sequence of uncorrelated Bernoulli variables (-1, 1).

    Page 94.

    """
    raise NotImplementedError()
