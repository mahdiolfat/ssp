"""Systems of state matrix representation from the Appendix."""


from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike


def convm(x: ArrayLike, p: int) -> np.ndarray:
    """Construct the convolution matrix of the signal x with p number of parameters.

    (N + p - 1) by p non-symmetric Toeplitz matrix
    """
    _x = np.array(x)
    if p < 1:
        raise ValueError(f"{p=} must be greater or equal to 1.")

    N = len(_x) + 2 * p - 2
    # the signal centered over its support
    # needed for the signal information-preserving frequency spectrum
    xcol = (_x.copy()).reshape(-1, 1)
    xpad = np.concatenate((np.zeros((p - 1, 1)), xcol, np.zeros((p - 1, 1))))
    X = np.empty([len(_x) + p - 1, p])
    for i in range(p):
        X[:, i] = xpad[p - i - 1:N - i, 0]
    return X


def covar(x: ArrayLike, p: int) -> np.ndarray:
    """Covariance Matrix.

    p x p hermitian toeplitz matrix of sample covariances
    """
    _x = np.array(x)
    m = len(_x)
    # remove the mean
    x0 = _x - np.mean(_x)
    R = np.transpose((convm(x0, p + 1).conjugate())) @ (convm(x0, p + 1) / (m - 1))
    return R


def normalprony(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Normalized prony Systems of Equations."""
    raise NotImplementedError()


def ywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Yuler-Walker Systems of Equations."""
    raise NotImplementedError()


def nywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Normalized Yuler-Walker Systems of Equations."""
    raise NotImplementedError()


def mywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Modified Yuler-Walker Systems of Equations."""
    raise NotImplementedError()


def eywe(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Extended Yuler-Walker Systems of Equations."""
    raise NotImplementedError()


def normaldeterministic(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Normal Determenistic Systems of Equations."""
    raise NotImplementedError()


def wienerhopf(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Wiener-Hopf Systems of Equations."""
    raise NotImplementedError()
