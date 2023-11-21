"""Test the state module."""


import numpy as np

from pyssp import state


def test_convm():
    x = np.array([1, 2, 3])
    p = 4

    expected = np.array([[1, 0, 0, 0],
                         [2, 1, 0, 0],
                         [3, 2, 1, 0],
                         [0, 3, 2, 1],
                         [0, 0, 3, 2],
                         [0, 0, 0, 3]], dtype=float)
    assert np.array_equal(expected, state.convm(x, p))


def test_covar():
    x = np.array([1, 2, 3])
    p = 4

    expected = np.array([[1, 0, -0.5, 0, 0],
                         [0, 1, 0, -0.5, 0],
                         [-0.5, 0, 1, 0, -0.5],
                         [0, -0.5, 0, 1, 0],
                         [0, 0, -0.5, 0, 1]], dtype=float)
    assert np.array_equal(expected, state.covar(x, p))
