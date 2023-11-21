"""Chapter 9 algorithm implementations."""

import numpy as np

from .state import convm


def lms(x, d, mu, nord, a0=None):
    '''LMS Adaptive Filter

    Refernce Page 506, Figure 9.7
    Suffers from gradient noise amplification
    '''
    X = convm(x, nord)
    M, N = X.shape

    if a0 is None:
        a0 = np.zeros(N)

    A = np.zeros((M - nord + 1))
    E = np.zeros((M - nord + 1))

    _a0 = np.array(a0)
    E[0] = d[0] - _a0 * X[0]
    A[0] = a0 + mu * E[0] * np.conjugate(X[0])

    for k in (1, M - nord + 1):
        E[k] = d[k] - A[k - 1] * X[k]
        A[k] = A[k - 1] + mu * E[k] * np.conjugate(X[k])

    return A, E


def nlms(x, d, beta, nord, a0):
    '''Normalized LMS Adaptive Filter

    Refernce Page 515, Figure 9.12
    '''

    delta = 0.0001
    X = convm(x, nord)
    M, N = X.shape

    if a0 is None:
        a0 = np.zeros(N)

    A = np.zeros((M - nord + 1))
    E = np.zeros((M - nord + 1))

    _a0 = np.array(a0)
    DEN = X[0] * X[0] + delta
    E[0] = d[0] - _a0 * X[0]
    A[0] = a0 + beta / DEN * E[0] * np.conjugate(X[0])

    # no further iterations if M == 1
    for k in (1, M - nord + 1):
        E[k] = d[k] - A[k - 1] * X[k]
        DEN = X[k] * X[k] + delta
        A[k] = A[k - 1] + beta / DEN * E[k] * np.conjugate(X[k])

    return A, E


def rls(x, d, nord, lamda=1, delta=0.001):
    '''Recursive Least Squares

    Reference 545, Table 9.6

    For special case of lambda == 1, this is known as growing window RLS algorithm,
    For all other values of lambda <1 and >0, this is the exponentially weighted RLS algorithm.
    '''

    X = convm(x, nord)
    M, N = X.shape
    P = np.eye(N) / delta
    W = np.zeros((M - nord + 1, N))

    for k in range(1, M - nord + 1):
        z = P * X[k]
        g = z / (lamda + X[k] * z)
        alpha = d[k] - X[k] @ np.transpose(W[k])
        W[k] = W[k - 1] + alpha * g
        P = (P - g * z) / lamda

    return W, P
