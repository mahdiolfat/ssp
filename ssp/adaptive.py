"""Chapter 9 algorithm implementations."""

import numpy as np
from numpy.typing import ArrayLike

from .state import convm


def lms(x: ArrayLike, d: ArrayLike, mu: int, nord: int,
        a0: None | ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """LMS Adaptive Filter.

    Refernce Page 506, Figure 9.7
    Suffers from gradient noise amplification
    """
    X = convm(x, nord)
    M, N = X.shape

    if a0 is None:
        a0 = np.zeros(N)

    A = np.zeros((M - nord + 1))
    E = np.zeros((M - nord + 1))

    _a0 = np.array(a0)
    _d = np.array(d)
    E[0] = _d[0] - _a0 * X[0]
    A[0] = a0 + mu * E[0] * np.conjugate(X[0])

    for k in (1, M - nord + 1):
        E[k] = _d[k] - A[k - 1] * X[k]
        A[k] = A[k - 1] + mu * E[k] * np.conjugate(X[k])

    return A, E


def nlms(x: ArrayLike, d: ArrayLike, beta: int, nord: int,
         a0: ArrayLike | None = None) -> tuple[ArrayLike, ArrayLike]:
    """Normalized LMS Adaptive Filter.

    Refernce Page 515, Figure 9.12
    """
    delta = 0.0001
    X = convm(x, nord)
    M, N = X.shape

    if a0 is None:
        a0 = np.zeros(N)

    A = np.zeros((M - nord + 1))
    E = np.zeros((M - nord + 1))

    _a0 = np.array(a0)
    _d = np.array(d)
    DEN = X[0] * X[0] + delta
    E[0] = _d[0] - _a0 * X[0]
    A[0] = a0 + beta / DEN * E[0] * np.conjugate(X[0])

    # no further iterations if M == 1
    for k in (1, M - nord + 1):
        E[k] = _d[k] - A[k - 1] * X[k]
        DEN = X[k] * X[k] + delta
        A[k] = A[k - 1] + beta / DEN * E[k] * np.conjugate(X[k])

    return A, E


def rls(x: ArrayLike, d: ArrayLike, nord: int, lamda: float=1,
        delta: float=0.001) -> tuple[ArrayLike, ArrayLike]:
    """Recursive Least Squares.

    Reference 545, Table 9.6

    For special case of lambda == 1, this is known as growing window RLS algorithm,
    For all other values of lambda <1 and >0, this is the exponentially weighted RLS algorithm.
    """
    _d = np.array(d)
    X = convm(x, nord)
    M, N = X.shape
    P = np.eye(N) / delta
    W = np.zeros((M - nord + 1, N))

    for k in range(1, M - nord + 1):
        z = P * X[k]
        g = z / (lamda + X[k] * z)
        alpha = _d[k] - X[k] @ np.transpose(W[k])
        W[k] = W[k - 1] + alpha * g
        P = (P - g * z) / lamda

    return W, P
