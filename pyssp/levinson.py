"""Chapter 5 algorithm implementations."""

import numpy as np


def rtoa(r) -> tuple[np.ndarray, np.ndarray]:
    """Recursively map a set of autocorrelations to a set of model parameters.

    The Levison-Durbin recursion.
    """
    a = np.ones((1, 1))
    epsilon = r[0]
    p = len(r) - 1
    r = r.reshape(-1, 1)

    for j in range(1, p + 1):
        gamma = -np.transpose(r[1:1 + j,]) @ np.flipud(a) / epsilon
        an = np.concatenate((a, np.zeros((1, 1)))).reshape(-1, 1)
        anT = np.conjugate(np.flipud(a))
        a = an + gamma * np.concatenate(([0], anT.ravel())).reshape(-1, 1)
        epsilon = epsilon * (1 - np.abs(gamma)**2)
        print(f"{gamma=},\n{a=},\n{epsilon=}\n")
    return a, epsilon


def gtoa(gamma):
    """Recursively map parameters to reflection coeffs.

    Reference Page 233, Table 5.2, Figure 5.6.

    Step up recursion defines how model parameters for a jth-order
    filter may be updated (stepped-up) to a (j + 1)st-order filter given reflection
    coefficients gamma.

    Cumulant generating function in statistics.
    """
    a = np.ones((1, 1))
    p = len(gamma)
    for j in range(1, p):
        a = np.concatenate((a, np.zeros((1, 1))))
        _a = a.copy()
        af = np.conjugate(np.flipud(_a))
        a = a + gamma[j] * af

    return a


def atog(a):
    """Recursively map from reflection coeffs to filter coeffs.

    The step-down recursion.

    Used within the framework of the "Shur-Cohn stability test".
    i.e., "the roots of the polynomial will lie inside the unit circle if and only
    if the magnitudes of the reflection coefficients are less than 1.
    i.e., the all-pole model/filter is minimum phase and guaranteed to be stable.

    Mapping from reflection coefficients to filter coefficients.
    """
    _a = np.array(a).reshape(-1, 1)
    p = len(_a)
    # drop a(0) and normalized in case it is not unity.
    _a = _a[1:] / _a[0]

    gamma = np.zeros((p - 1, 1))
    gamma[p - 2] = _a[p - 2]

    for j in range(p - 2, 0, -1):
        # print(f"{gamma=}, {_a=}")
        ai1 = _a[:j].copy()
        ai2 = _a[:j].copy()
        af = np.flipud(np.conjugate(ai1))
        # print(f"{ai1=}, {ai2=}, {af=}")
        s1 = ai2 - gamma[j] * af
        s2 = 1 - np.abs(gamma[j])**2
        _a = np.divide(s1, s2)
        # print(f"{s1=}, {s2=}, {_a=}")
        gamma[j - 1] = _a[j - 1]

    return gamma


def gtor(gamma, epsilon=None):
    """Find the autocorrelation sequence from the reflection coefficients and the modeling error.

    Page 241, Figure 5.9.
    """
    p = len(gamma)
    aa = np.array([[gamma[0]]]).reshape(-1, 1)
    r = np.array(([1, -gamma[0]])).reshape(1, -1)

    for j in range(1, p):
        aa1 = np.concatenate((np.ones((1, 1)), aa)).reshape(-1, 1)
        aa0 = np.concatenate((aa, np.zeros((1, 1)))).reshape(-1, 1)
        aaf = np.conjugate(np.flipud(aa1))
        aa = aa0 + gamma[j] * aaf
        print(aa)
        rf = -np.fliplr(r) @ aa
        print(rf)
        print(rf.shape)
        print(r)
        print(r.shape)
        print()
        r = np.concatenate((r[0], rf[0])).reshape(1, -1)

    if epsilon is not None:
        r = r * epsilon / np.prod(1 - np.abs(gamma)**2)

    return r


def ator(a, b):
    """Page 241, Figure 5.9."""
    gamma = atog(a)
    r = gtor(gamma.ravel())
    r = r * np.sqrt(b) / np.prod(1 - np.abs(gamma)**2)

    return r


def rtog(r):
    """Recurse from the model params to filter coeffs.

    The Shur Recursion: Table 5.5.

    Implementation based on Figure A.3, page 581
    """
    a, _ = rtoa(r)
    gamma = atog(a)
    return gamma


def glev(r, b):
    """General Levinson Recursion, solves any Hermitian Toeplitz matrix.

    Can solve the Wiener-Hopf system of equations for Optimal MSE Filter design.
    """
    _r = np.array(r).reshape(-1, 1)
    # _b = np.array([b]).reshape(-1, 1)
    p = len(b)
    a = np.array([[1]]).reshape(-1, 1)
    x = np.array([b[0] / r[0]]).reshape(-1, 1)
    epsilon = r[0]
    for j in range(1, p):
        print(j)
        print(f"{_r=}, {_r.shape=}")
        _r1 = np.transpose(np.array(_r[1:j + 1]))
        print(f"{_r1=}, {_r1.shape=}")
        print(f"{x=}, {x.shape=}")
        print(f"{a=}, {a.shape=}")
        g = _r1 @ np.flipud(a)
        print(f"{g=}, {g.shape=}")
        gamma = -g / epsilon
        print(f"{gamma=}, {gamma.shape=}")
        _a0 = np.concatenate([a, [[0]]])
        _af = np.conjugate(np.flipud(_a0))
        print(f"{_a0=}, {_a0.shape=}")
        print(f"{_af=}, {_af.shape=}")
        a = _a0 + gamma * _af
        epsilon = epsilon * (1 - np.abs(gamma)**2)
        print(f"{epsilon=}")
        delta = _r1 @ np.flipud(x)
        q = (b[j] - delta[0, 0]) / epsilon
        x = np.concatenate([x, [[0]]])
        x = x + q * np.conjugate(np.flipud(a))
        print()

    return x
