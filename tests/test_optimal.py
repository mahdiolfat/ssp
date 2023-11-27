"""Validation implementation of optimal controllers/estimators."""

import numpy as np
from pyssp import optimal


def test_kalman() -> None:
    """Validate Kalman filter (estimator) implementation."""

    av = 1
    aw = 0.36
    A = 0.8
    C = 1
    N = 10

    rng = np.random.default_rng(seed=100)

    # because the assumption is that vn and wn (noise in observations and system) must be draw
    # independently from two seperate random processes
    rw, rv = rng.spawn(2)
    wn = rw.normal(0, aw, N)
    vn = rv.normal(0, av, N)

    xn = 0.8 * np.ones(N) + wn
    yn = xn + vn

    P0, P1, K, xhat0, xhat1 = optimal.kalman(yn, A, C, aw, av)
