"""Systems of state matrix representation from the Appendix."""


import numpy as np


def convm(x, p):
    """Convolution Matrix
    (N + p - 1) by p non-symmetric Toeplitz matrix
    """

    if p < 1:
        raise ValueError(f"{p=} must be greater or equal to 1.")

    N = len(x) + 2 * p - 2
    # the signal centered over its support
    # needed for the signal information-preserving frequency spectrum
    xcol = (x.copy()).reshape(-1, 1)
    xpad = np.concatenate((np.zeros((p - 1, 1)), xcol, np.zeros((p - 1, 1))))
    X = np.empty([len(x) + p - 1, p])
    for i in range(p):
        X[:, i] = xpad[p - i - 1:N - i, 0]
    return X


def covar(x, p):
    '''Covariance Matrix
    p x p hermitian toeplitz matrix of sample covariances
    '''

    m = len(x)
    # remove the mean
    x0 = x.copy() - np.mean(x)
    R = np.transpose((convm(x0, p + 1).conjugate())) @ (convm(x0, p + 1) / (m - 1))
    return R


# pylama -i W0613
def normalprony(x, p, q):
    """Normal Prony Systems of Equations"""


# pylama -i W0613
def ywe(x, p, q):
    """Yuler-Walker Systems of Equations"""


# pylama -i W0613
def nywe(x, p, q):
    """Normalized Yuler-Walker Systems of Equations"""


# pylama -i W0613
def mywe(x, p, q):
    """Modified Yuler-Walker Systems of Equations"""


# pylama:ignore=W0613
def eywe(x, p, q):
    """Extended Yuler-Walker Systems of Equations"""


def normaldeterministic():
    """Normal Determenistic Systems of Equations"""


def wienerhopf():
    """Wiener-Hopf Systems of Equations"""
