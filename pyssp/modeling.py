"""Chapter 4 modeling algorithm implementations."""

import logging
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from .state import convm

logger = logging.getLogger(__name__)


def pade(x: ArrayLike, p: int, q: int) -> tuple[ArrayLike, ArrayLike]:
    """Reference Page 138, Table 4.1.

    The Pade approximation models a signal as the unis sample response
    of linear shift invariant system have p poles and q zeros.
    """
    _x = np.array(x).reshape(-1)
    if p + q > len(_x):
        raise ValueError(f"Model order {p + q} is too large.")

    X = convm(_x, p + 1)

    # Linear difference matrix spanning the number of zeros
    Xq = X[q + 1:q + p + 1, 1:p + 1].copy()
    a = np.linalg.solve(-Xq, X[q + 1: q + p + 1, 0])
    # a(0) normalized to 1
    a = np.concatenate((np.ones(1), a)).reshape(-1, 1)
    b = X[:q + 1, :p + 1] @ a

    return a.ravel(), b.ravel()


def prony(x: ArrayLike, p: int, q: int) -> tuple[ArrayLike, ArrayLike, float]:
    """Least square minimization of poles to get denominator coefficients.

    Solves directly (Pade method) to get numerator coefficients.
    Also calculates minimum error achieved.

    Condition to energy_match is on page 575
    """
    _x = np.array(x).reshape(-1, 1)
    N = len(_x)
    if p + q > N:
        raise ValueError(f"Model order {p + q} is too large.")

    X = convm(x, p + 1)

    # the factorization does not guarantee nonsingularity!
    # resulting matrix is positive *semi*-definite: all zeros are
    # on/inside the unit circle
    Xq = X[q:N + p - 1, :p].copy()
    Xq1 = X[q + 1:N + p, 0].copy()
    Xq_H = Xq.conjugate().transpose()
    rx = Xq_H @ Xq1
    Xinv = np.linalg.inv(Xq_H @ Xq)
    a = -Xinv @ rx
    # a(0) normalized to 1
    a = np.concatenate((np.ones(1), a)).reshape(-1, 1)
    # same as Pade method
    b = X[:q + 1, :p + 1] @ a

    # the minimum squared error
    err = np.transpose(_x[q + 1:N]) @ X[q + 1:N, :p + 1] @ a

    return a.ravel(), b.ravel(), err.ravel()[0]


def shanks(x: ArrayLike, p: int, q: int) -> tuple[ArrayLike, ArrayLike, float]:
    """Shank's method."""
    _x = np.array(x).ravel().reshape(-1, 1)
    N = len(_x)
    if p + q >= N:
        raise ValueError(f"Model order {p + q} is too large.")

    a, _, _ = prony(x, p, q)
    logger.info(f"{a=}")
    u = np.concatenate((np.ones(1), np.zeros(N - 1)))
    sos = signal.tf2sos([1], a)
    g = signal.sosfilt(sos, x=u)
    logger.info(f"{g=}")
    G = convm(g, q + 1)
    G0 = G[:N].copy()
    logger.info(f"{G0=}")
    b = np.linalg.lstsq(G0, _x, rcond=None)[0]
    err = _x.T @ _x - _x.T @ G[:N, :q + 1] @ b

    return a, b.ravel(), err.ravel()[0]


def spike(g: ArrayLike, n0: int, n: int) -> tuple[ArrayLike, float]:
    """Leaset Squares Inverse Filter."""
    _g = np.array(g).reshape(-1, 1)
    m = len(_g)

    if m + n - 1 <= n0:
        raise ValueError(f"m + n - 1 must be less than {n0=}")

    G = convm(g, n)
    d = np.zeros((m + n - 1, 1))
    d[n0] = 1
    h = np.linalg.lstsq(G, d, rcond=None)[0]
    err = 1 - G[n0,] @ h

    return h.ravel(), err.ravel()[0]


def ipf(x: ArrayLike, p: int, q: int, n: None, a: ArrayLike) -> NoReturn:
    """Iterative Pre-Filtering."""
    raise NotImplementedError()


def acm(x: ArrayLike, p: int) -> tuple[np.ndarray, float]:
    """The auto-correlation method."""
    x0 = np.array(x).ravel().reshape(-1, 1)
    N = len(x0)
    if p >= len(x0):
        raise ValueError("p (all-pole model) too large")

    X = convm(x, p + 1)
    Xq = X[:N + p - 1, :p].copy()
    rx = X[1:N + p, 0].copy()
    Xq_H = Xq.copy().conjugate().transpose()
    Xinv = np.linalg.inv(Xq_H @ Xq)
    a1 = -Xinv @ Xq_H @ rx
    a = np.concatenate((np.ones(1), a1)).reshape(-1, 1)
    err = np.abs(X[:N + p, 0].T @ X @ a)

    return a, err.ravel()[0]


def covm(x: ArrayLike, p: int)-> tuple[ArrayLike, float]:
    """Solve the complete Prony normal equations."""
    _x = np.array(x).ravel().reshape(-1, 1)
    N = len(_x)
    if p >= N:
        raise ValueError(f"{p=} all-pole model too large")

    X = convm(_x, p + 1)
    Xq = X[p - 1:N - 1, :p].copy()
    Xsol = np.linalg.lstsq(-Xq, X[p:N, 0], rcond=None)[0]
    logger.info(f"{Xsol=}")
    a = np.hstack(([1], Xsol))
    err = np.abs(X[p:N,0] @ X[p:N,] @ a)
    return a, err.ravel()[0]


def durbin(x: ArrayLike, p: int, q: int) -> tuple[ArrayLike, ArrayLike]:
    """The durbin method."""
    _x = np.array(x).ravel().reshape(-1, 1)
    N = len(_x)
    if p >= N:
        raise ValueError("p (all-pole model) too large")

    a, eps = acm(_x, p)
    b, eps = acm(a / np.sqrt(eps), q)
    b = b / np.sqrt(eps)
    return a, b


def mywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Modified Yuler-Walker Systems of Equations.

    Page 190.
    """
    raise NotImplementedError()


def eywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Extended Yuler-Walker Systems of Equations.

    Page 193.
    """
    raise NotImplementedError()
