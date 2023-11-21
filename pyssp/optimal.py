"""Optimal Wiener Filters."""


import numpy as np
import numpy.typing as npt


def kalman(y: list[float], A: npt.ArrayLike, C: npt.ArrayLike, sigmaw: list[float],
           sigmav: list[float]) -> tuple[npt.NDArray, ...]:
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


def wiener_denoise() -> None:
    """Denoising based on IIR wiener filters."""
    raise NotImplementedError()


def wiener_systemid() -> None:
    """Systemid based on FIR wiener filters."""
    raise NotImplementedError()
