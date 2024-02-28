"""Test validation of levinson based recursion routines."""

import logging

import numpy as np
from ssp import levinson


logger = logging.getLogger(__name__)


def test_glev():
    '''Example 5.3.1, Page 266'''
    r = [4, 2, 1]
    b = [9, 6, 12]

    expected = [2, -1, 3]

    sol = levinson.glev(r, b)
    logger.info(sol)

    assert np.allclose(sol, expected)


def test_gtor() -> None:
    '''Based on example 5.2.6'''
    expected_rx = np.array([2, -1, -1/4, 1/8])

    gamma = [1/2, 1/2, 1/2]
    epsilon = 2 * (3 / 4)**3
    rx = levinson.gtor(gamma, epsilon)
    assert np.allclose(rx, expected_rx)


def test_atog() -> None:

    a = [1, 0.5, -0.1, -0.5]
    expected_g = np.array([0.5, 0.2, -0.5])

    gamma = levinson.atog(a)
    logger.info(gamma)
    logger.info(expected_g)

    assert np.allclose(gamma, expected_g)


def test_rtog() -> None:
    rx = [2, -1, -1/4, 1/8]
    expected_g = np.array([0.5, 0.5, 0.5])

    gamma = levinson.rtog(rx)
    logger.info(gamma)
    logger.info(expected_g)

    assert np.allclose(gamma, expected_g)


def test_ator() -> None:
    a = [1, 1, 7/8, 1/2]
    epsilon = 2 * (3 / 4)**3
    b = epsilon**2
    expected_rx = np.array([2, -1, -1/4, 1/8])

    rx = levinson.ator(a, b = b)
    logger.info(rx)
    logger.info(expected_rx)

    assert np.array_equal(rx, expected_rx)


def test_gtoa() -> None:
    gamma = [0.5, 0.2, -0.5]
    expected_a = np.array([1, 0.5, -0.1, -0.5])

    a = levinson.gtoa(gamma)
    logger.info(a)
    logger.info(expected_a)

    assert np.allclose(a, expected_a)

def test_rtoa() -> None:
    rx = np.array([2, -1, -1/4, 1/8])
    expected_a = [1, 1, 7/8, 1/2]
    expected_eps = 2 * (3 / 4)**3

    a, eps = levinson.rtoa(rx)

    logger.info(f"{a=}")
    logger.info(f"{expected_a=}")
    logger.info(f"{eps=}")
    logger.info(f"{expected_eps=}")

    assert np.allclose(a, expected_a) and np.allclose(eps, expected_eps)
