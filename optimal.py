"""Optimal Wiener Filters."""


import numpy as np


def kalman(y, A, C, sigmaw, sigmav, u=None):
    """Kalman Filter.

    y: vector of observations N x q,  n time steps, q sensors
    A: time-varying N x (p x p) state transition matrix
    C: time-varying N x (q x p) measurement transformations
    sigmaw: time-varying process white noise variance(s) (1-D time-ordered list of variances)
    sigmav: time-varying measurement white noise variance(s) (1-D time-ordered list of variances)
    """

    _y = np.array(y, ndmin=2)
    _A = np.array(A, ndmin=2)
    _C = np.array(C, ndmin=2)
    Qw = np.diag(np.array(sigmaw, ndmin=1))
    Qv = np.diag(np.array(sigmav, ndmin=1))
    N = np.shape(_y)[1]
    print(_y)

    p = np.shape(_A)[0]
    q = np.shape(_C)[0]
    print(f"{p}, {q}, {N=}")

    # The a priori error covariance matrix
    P0 = np.zeros((N + 1, p, p))
    # The a priori error (unbiased) estimate
    xhat0 = np.zeros((N + 1, p))

    # The a posteriori error covariance matrix
    P1 = np.zeros((N + 1, p, p))
    # The a posteriori estimate (linear prediction)
    xhat1 = np.zeros((N + 1, p))

    # System dynamical model based on given transitions and sensor transformations
    # No dependance on noise of measurements

    # Time varying N x p Kalman Gain
    K = np.zeros((N + 1, p, q))

    # setting up initial values states before recursion
    xhat1[0] = 0
    P1[0] = 1
    xhat0[0] = 0
    P0[0] = 1

    C_H = np.conjugate(np.transpose(_C))
    A_H = np.conjugate(np.transpose(_A))

    for n in range(N):
        xhat0[n] = _A @ xhat1[n]
        P0[n] = _A @ P1[n] @ A_H + Qw

        Fisher = _C @ P0[n] @ C_H + Qv
        K[n] = P0[n] @ C_H @ np.linalg.inv(Fisher)

        xhat1[n + 1] = xhat0[n] + K[n] @ (_y[0, n] - _C @ xhat0[n])
        P1[n + 1] = (np.eye(p) - K[n] @ _C) @ P0[n]

    return P0, P1, K, xhat0, xhat1


def wiener_denoise():
    pass


def wiener_systemid():
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    av = 1
    aw = 0.36
    A = 0.8
    C = 1
    N = 10

    rng = np.random.default_rng(seed=100)

    rw, rv = rng.spawn(2)

    wn = rw.normal(0, aw, N)
    vn = rv.normal(0, av, N)

    xn = 0.8 * np.ones(N) + wn
    yn = xn + vn

    P0, P1, K, xhat0, xhat1 = kalman(yn, A, C, aw, av)

    plt.plot(xn, label='x')
    plt.plot(yn, label='y')
    plt.plot(xhat0, label='xhat0')
    plt.plot(xhat1, label='xhat1')

    plt.legend()
    plt.show()
