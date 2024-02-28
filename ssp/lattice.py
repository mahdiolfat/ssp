"""Implementation of algorithm from Chapter 6."""

import logging
from typing import NoReturn

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def fcov(x: ArrayLike, p: int) -> tuple[ArrayLike, ArrayLike]:
    """Figure 6.15, Page 310.

    Using the forward covariance method the reflection co-efficients of the lattice filter
    are found by sequentially minimizing the sum of the squares of the forward prediction error.
    """
    _x = np.array(x)
    if p >= len(_x):
        raise ValueError("Model order must be less than length of signal")

    _x = np.array(_x).reshape(-1, 1)
    N = len(_x)
    eplus = _x[1:N]
    eminus = _x[:N - 1]

    gamma = np.empty((p, 1))
    err = np.empty((p, 1))

    for j in range(p):
        logger.info(j)
        N = N - 1
        logger.info(f"{eplus=}, {eplus.shape=}")
        logger.info(f"{eminus=}, {eminus.shape=}")
        gamma[j] = (np.transpose(-eminus) @ eplus) / (np.transpose(eminus) @ eminus)
        temp1 = eplus + gamma[j] * eminus
        temp2 = eminus + np.conjugate(gamma[j]) * eplus
        err[j] = np.transpose(temp1) @ temp1
        eplus = temp1[1:N]
        eminus = temp2[:N - 1]
        logger.info(gamma)
        logger.info(err)

    return gamma, err


def burg(x: ArrayLike, p: int) -> tuple[ArrayLike, ArrayLike]:
    """Sequentially minimizes the sum of the forward and backward covariance errors.

    Guaranteed to be stable. All reflection coefficients will be <|1|
    """
    _x = np.array(x).reshape(-1, 1)
    N = len(_x)
    if p > N:
        raise ValueError("Model order must be less than length of signal")

    eplus = _x[1:N]
    eminus = _x[:N - 1]

    gamma = np.empty((p, 1))
    err = np.empty((p, 1))

    for j in range(p):
        logger.info(j)
        N = N - 1
        logger.info(f"{eplus=}, {eplus.shape=}")
        logger.info(f"{eminus=}, {eminus.shape=}")
        eplusmag = np.transpose(eplus) @ eplus
        eminusmag = np.transpose(eplus) @ eplus
        gamma[j] = (np.transpose(-2 * eminus) @ eplus) / (eplusmag + eminusmag)
        temp1 = eplus + gamma[j] * eminus
        temp2 = eminus + np.conjugate(gamma[j]) * eplus
        err[j] = np.transpose(temp1) @ temp1 + np.transpose(temp2) @ temp2
        eplus = temp1[1:N]
        eminus = temp2[:N - 1]

    return gamma, err


def bcov(x: ArrayLike, p: int) -> NoReturn:
    """Sequentially minimizes the backward covariance error."""
    raise NotImplementedError()


def mcov(x: ArrayLike, p: int) -> tuple[ArrayLike, ArrayLike]:
    """Modified covariance method.

    Unlike the forward/backward algorithms.
    It *does not* minimize an error term sequentially.
    """
    _x = np.array(x).reshape(-1, 1)
    N = len(_x)

    if p >= len(_x):
        raise ValueError("Model order must be less than length of signal")

    X = sp.linalg.toeplitz(_x[p:N], np.flipud(_x[:p + 1]))
    R = np.transpose(X) @ X
    R1 = np.array(R[1:p + 1, 1: p + 1])
    R2 = np.array(np.flipud(np.fliplr(R[:p, :p])))
    b1 = np.array(R[1:p + 1, 1])
    b2 = np.array(np.flipud(R[:p, p]))

    Rx = -R1 - R2
    b = b1 + b2
    a = sp.linalg.solve_toeplitz(Rx[:, 1], b)
    a = np.concatenate(([1], a))
    err = np.dot(R[0], a) + np.dot(np.flip(R[p]), a)

    return a, err
