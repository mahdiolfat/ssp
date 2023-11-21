import numpy as np
import scipy as sp

from state import convm


def pade(x, p, q):
    '''
    Reference Page 138, Table 4.1
    The Pade approximation models a signal as the unis sample response
    of linear shift invariant system have p poles and q zeros.
    '''
    if p + q > len(x):
        raise ValueError(f"Model order {p + q} is too large.")

    X = convm(x, p + 1)

    # Linear difference matrix spanning the number of zeros
    Xq = X[q + 1:q + p + 1, 1:p + 1].copy()
    print(Xq.shape)
    a = np.linalg.solve(-Xq, X[q + 1: q + p + 1, 0])
    # a(0) normalized to 1
    a = np.concatenate((np.ones(1), a)).reshape(-1, 1)
    b = X[:q + 1, :p + 1] @ a

    return (a, b)


def prony(x, p, q):
    '''
    Least square minimization of poles to get denominator coefficients.
    Solves directly (Pade method) to get numerator coefficients.
    Also calculates minimum error achieved.

    Condition to energy_match is on page 575
    '''

    if p + q > len(x):
        raise ValueError(f"Model order {p + q} is too large.")

    # copy and make given signal column array
    X = convm(x, p + 1)
    print(X.shape)
    # M = p + q
    N = len(x)
    print(f"{N=}")
    xc = x.copy().reshape(-1, 1)

    # Xq = X[q + 1:q + p + 1, 1:p + 1].copy()
    # a = np.linalg.solve(-Xq, X[q + 1: q + p + 1, 0])
    # a = np.concatenate((np.ones(1), a)).reshape(-1, 1)
    # b = X[:q + 1, :p + 1] @ a

    # the factorization does not guarantee nonsingularity!
    # resulting matrix is positive *semi*-definite: all zeros are
    # on/inside the unit circle
    Xq = X[q:N + p - 1, :p].copy()
    Xq1 = X[q + 1:N + p, 0].copy()
    Xq_H = Xq.conjugate().transpose()
    rx = Xq_H @ Xq1
    Xinv = np.linalg.inv(Xq_H @ Xq)
    a = -Xinv @ rx
    print(a.shape)
    # a(0) normalized to 1
    a = np.concatenate((np.ones(1), a)).reshape(-1, 1)
    # same as Pade method
    b = X[:q + 1, :p + 1] @ a

    # the minimum squared error
    err = np.transpose(xc[q + 1:N]) @ X[q + 1:N, :p + 1] @ a

    return a, b, err


def shanks(x, p, q):
    '''
    '''

    N = len(x)
    if p + q >= N:
        raise ValueError(f"Model order {p + q} is too large.")

    a, _, _ = prony(x, p, q)
    print(f"{a.transpose().ravel()=}")
    u = np.concatenate((np.ones(1), np.zeros(N - 1)))
    zpk = sp.signal.tf2zpk([1], a.ravel())
    sos = sp.signal.zpk2sos(*zpk)
    res = sp.signal.sosfilt(sos, x=u)
    G = convm(res.ravel(), q + 1)
    G0 = G[:N,].copy()
    print(f"{G0.shape=}")
    G0_H = np.transpose((G0.copy()).conjugate())
    x0 = (x.copy()).reshape(-1, 1)
    gx = G0_H @ x0
    # the factorization does not guarantee nonsingularity!
    # resulting matrix is positive *semi*-definite
    Ginv = np.linalg.inv(G0_H @ G0)
    print(f"{x.shape=}")
    print(f"{Ginv.shape=}")
    b = Ginv @ gx
    err = 1

    return a, b, err


def spike(g, n0, n):
    '''Leaset Squares Inverse Filter'''

    g = g.reshape(-1, 1)
    m = len(g)

    if m + n - 1 <= n0:
        raise ValueError(f"m + n - 1 must be less than {n0=}")

    G = convm(g, n)
    d = np.zeros((m + n - 1, 1))
    d[n0] = 1
    print(f"{d.shape=}, {G.shape=}")
    G_H = np.transpose(G.conjugate())

    print(f"{G_H.shape=}, {G.shape=}")
    Ginv = np.linalg.inv(G_H @ G)
    h = Ginv @ G_H @ d

    return h


def ipf(x, p, q, n=10, a=None):
    pass


def acm(x, p) -> tuple[np.ndarray, np.ndarray]:
    x0 = x.copy().ravel().reshape(-1, 1)
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

    return a, err


def covm(x, p):
    '''
    Solve the complete Prony normal equations.
    '''
    x0 = x.copy().ravel().reshape(-1, 1)
    N = len(x0)
    if p >= len(x0):
        raise ValueError("p (all-pole model) too large")

    X = convm(x0, p + 1)
    Xq = X[p - 1:N - 1, :p].copy()
    cx = X[p:N, 0].copy()
    Xq_H = Xq.copy().conjugate().transpose()
    print(f"{Xq=}")
    Xinv = np.linalg.inv(Xq_H @ Xq)
    a1 = -Xinv @ Xq_H @ cx
    a = np.concatenate((np.ones(1), a1)).reshape(-1, 1)
    err = np.abs(cx.transpose() @ X[p:N,] @ a)
    return a, err


def durbin(x, p, q):
    x0 = x.copy().ravel().reshape(-1, 1)
    # N = len(x0)
    if p >= len(x0):
        raise ValueError("p (all-pole model) too large")

    a, eps = acm(x, p)
    b, eps = acm(a / np.sqrt(eps), q)
    b /= np.sqrt(eps)
    return a, b
