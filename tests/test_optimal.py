"""Test optimal."""

import numpy as np

from ssp import optimal


def test_kalman():
    """Discrete Kalman filter."""
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

    P0, P1, K, xhat0, xhat1 = optimal.kalman(yn, A, C, aw, av)
