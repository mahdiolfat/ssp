"""Power spectrum and Frequency Estimators."""

from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike

from .modeling import acm
from .state import covar


def periodogram(x: ArrayLike, n1: int=0, n2: None | int=None, nfft: None | int=1024) -> np.ndarray:
    """Periodogram, non-paramteric spectrum estimator.

    Ergodic in signal length converging to the true power spectrum.

    Reference Page 394, Figure 8.1
    """
    _x = np.array(x)

    if n2 is None:
        n2 = len(_x)

    if nfft is None:
        nfft = max(1024, n2 + 1)

    nfft = max(nfft, len(_x))
    Px = np.abs(np.fft.fft(_x[n1:n2], nfft)) ** 2 / (n2 - n1)

    # DC carries no information
    Px[0] = Px[1]

    return Px


def mper(x: ArrayLike, n1: int, n2: int, win: ArrayLike | None) -> np.ndarray:
    """Estimates non-paramteric spectrum estimator.

    Same as the periodogram but the signal is windowed first.
    Spectral smoothing via time-domain windowing.

    Reference Page 410, Figure 8.11
    """
    _x = np.array(x)
    N = n2 - n1

    # rectangular window
    if win is None:
        win = np.ones(N)
    xw = _x[n1:n2] * win / np.linalg.norm(win)
    Px = N * periodogram(xw)

    return Px


def bart(x: ArrayLike, nsect: int) -> np.ndarray:
    """Bartlett's method non-paramteric spectrum estimation.

    Reference Page 414, Figure 8.13.
    Averages periodograms of non-overlapping sections.
    """
    _x = np.array(x).ravel().reshape(-1, 1)
    L = len(_x) // nsect
    Px = np.zeros(1)

    n1 = 0
    for _ in range(1, nsect + 1):
        Px = Px + periodogram((_x[n1:n1 + L - 1])) / nsect
        n1 = n1 + L

    return Px


def welch(x: ArrayLike, L: int, over: int, win: ArrayLike | None) -> np.ndarray:
    """Welch's method: non-paramteric spectrum estimator.

    Reference Page 418, Figure 8.16
    """
    _x = np.array(x)
    if over >= 1 or over < 0:
        raise ValueError(f"Overlap {over=} is invalid")

    if win is None:
        win = np.ones(len(_x))

    n1 = 0
    n0 = (1 - over) * L

    nsect = 1 + (len(_x) - L) // n0
    Px = np.zeros(1)
    for _ in range(nsect):
        Px = Px + mper(_x, n1, n1 + L - 1, win)
        n1 = n1 + n0

    return Px


def per_smooth(x: ArrayLike, M: int, n1: int=0, n2: None=None, win: None=None) -> np.ndarray:
    """Blackman-Tukey Spectrum Estimator: non-parametric estimation.

    Page 421, Figure 8.18
    """
    _x = np.array(x)

    _n2: int
    if n2 is None:
        _n2 = len(_x)

    R = np.array(covar(_x[n1:n2], M))
    r = np.concatenate(np.array((np.flip(R[0, 1:M]), [R[0, 0]], R[0, 1:M])))

    _win: ArrayLike
    if win is None:
        _win = np.ones(M)
    r = r * _win
    nfft = max(1024, M)
    Px = np.abs(np.fft.fft(r, nfft))
    Px[0] = Px[1]

    return Px


def minvar(x: ArrayLike, p: int) -> np.ndarray:
    """Minimum Variance spectrum estimator: parametric estimation.

    Page 430, Figure 8.19
    """
    _x = np.array(x)
    R = covar(x, p)
    d, v = np.linalg.eig(R)
    U = np.diag(np.linalg.inv(np.abs(d) + np.finfo(float).eps))
    nfft = max(1024, len(_x) + 1)
    V = np.abs(np.fft.fft(v, nfft))**2
    Px = 10 * np.log10(p) - 10 * np.log10(V @ U)

    return Px


def mem(x: ArrayLike, p: int) -> np.ndarray:
    """Maximum Entropy spectrum estimator: parametric estimation.

    Page 437, Figure 8.19
    """
    a, e = acm(x, p)
    nfft = max(len(a) + 1, 1024)
    Px = 20 * (np.log10(e) - np.log10(np.abs(np.fft.fft(a, nfft))))

    return Px


def modal(x: ArrayLike, p: int, q: int) -> NoReturn:
    """Mode-based spectrum estimator: parametric estimation.

    Page 440.

    Arguements: (x, p, q)
    """
    raise NotImplementedError()


def phd(x: ArrayLike, p: int) -> tuple[np.ndarray, float]:
    """Pisarenko Harmonic Decomposition frequency estimator.

    A noise subspace method.

    Page 461, Figure 8.33
    """
    _x = np.array(x)
    R = covar(_x, p + 1)
    d, v = np.linalg.eig(R)
    ddiag = np.diag(d)
    index = np.argmin(ddiag)
    sigma = ddiag[index]
    vmin = v[:, index]

    return vmin, sigma


def music(x: ArrayLike, p: int, M: int) -> ArrayLike:
    """MUSIC, Frequency estimator.

    A noise subspace method.

    Page 430, Figure 8.19
    """
    _x = np.array(x).ravel().reshape(-1, 1)
    if p + 1 > M or len(_x) < M:
        raise ValueError("Size of signal covariance matrix is inappropriate.")

    R = covar(x, M)
    d, v = np.linalg.eig(R)
    ddiag = np.diag(d)
    i = np.argsort(ddiag)
    # y = ddiag[i]
    Px = np.zeros(1, dtype=float)

    nfft = max(len(_x) + 1, 1024)
    for j in range(M - p):
        Px = Px + np.abs(np.fft.fft(v[:, i[j]], nfft))

    Px = -20 * np.log10(Px)

    return Px


def ev(x: ArrayLike, p: int, M: int) -> np.ndarray:
    """Eigenvector spectrum estimator: noise subspace method, parametric.

    A noise subspace method.

    Reference Page 466, Figure 8.35
    """
    if p + 1 > M:
        raise ValueError('Specified signal size is too small')

    _x = np.array(x)
    R = covar(_x, M)
    d, v = np.linalg.eig(R)
    # ddiag = np.diag(d)
    yi = np.argsort(np.diag(d))
    # y = ddiag[yi]
    Px = np.zeros(1, dtype=float)
    nfft = max(1024, M + p + 1)
    for j in range(M - p):
        Px = Px + np.abs(np.fft.fft(v[:, yi[j]], nfft))

    Px = -10 * np.log10(Px)

    return Px


def min_norm(x: ArrayLike, p: int, M: int) -> np.ndarray:
    """Minimum Norm spectrum estimator: noise subspace method, parametric.

    Frequency estimator.
    Reference Page 466, Figure 8.35
    """
    _x = np.array(x)
    if p + 1 > M:
        raise ValueError('Specified signal size is too small')

    R = covar(_x, M)
    d, v = np.linalg.eig(R)
    yi = np.argsort(np.diag(d))

    V = np.empty((1, M - p))
    for j in range(M - p):
        V[j] = v[:, yi[j]]

    a = V @ np.transpose(V.conjugate())

    nfft = max(1024, M + p + 1)
    Px = -20 * np.log10(np.abs(np.fft.fft(a, nfft)))

    return Px


def bt_pc(x: ArrayLike, p: int, M: int) -> np.ndarray:
    """Blackman-Tukey Principle-Component Frequeny Estimator.

    program for estimating the frequencies of pcomplex exponentials in white noise using
    a principal components analysis with the Blackman-Tukey method.

    Refence Page 471, Figure 8.38
    """
    _x = np.array(x)
    if p + 1 > M:
        raise ValueError('Specified signal size is too small')

    R = covar(_x, M)
    d, v = np.linalg.eig(R)
    ddiag = np.diag(d)
    yi = np.argsort(np.diag(d))
    y = ddiag[yi]
    Px = np.zeros(1)
    nfft = max(1024, M)
    for j in range(M - p, M):
        Px = Px + np.abs(np.fft.fft(v[:, yi[j]], nfft)) * np.sqrt(np.real(y[j]))

    Px = 20 * np.log10(Px) - 10 * np.log10(M)

    return Px


def mv_pc(x: ArrayLike, p: int, M: int) -> NoReturn:
    """Minimum Variance, Principle Component."""
    raise NotImplementedError()


def ar_pc(x: ArrayLike, p: int, M: int) -> NoReturn:
    """Autoregressive, Principle Component."""
    raise NotImplementedError()


def overlay(N: int, omega: int, A: ArrayLike, sigma: int, num: int) -> NoReturn:
    """Periodogram overlays: using an ensemble of realizations.

    Summary of this function.
    Arguements: (N, omega, A, sigma, num)
    """
    raise NotImplementedError()
