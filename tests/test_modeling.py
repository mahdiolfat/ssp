"""Tests for stochastic models."""

import logging

import numpy as np
import scipy.signal as signal
from pyssp import modeling, system


logger = logging.getLogger(__name__)


def test_pade() -> None:
    x = [1, 1.5, 0.75, 0.1875, 0.0938]
    expected_a = [1, -1.5, 1.5]
    expected_b = [1]

    a, b = modeling.pade(x, p = 2, q = 0)

    assert np.array_equal(a, expected_a) and np.array_equal(b, expected_b)

    expected_a = [1]
    expected_b = [1, 1.5, 0.75]

    a, b = modeling.pade(x, p = 0, q = 2)
    logger.info(a)
    logger.info(b)

    assert np.array_equal(a, expected_a) and np.array_equal(b, expected_b)

    expected_a = [1, -0.5]
    expected_b = [1, 1]

    a, b = modeling.pade(x, p = 1, q = 1)
    logger.info(a)
    logger.info(b)

    assert np.array_equal(a, expected_a) and np.array_equal(b, expected_b)


def test_prony():
    N = 21
    T = 2 * (N - 1) + 1
    xn = np.ones(T)
    xn[N:] = 0

    res = modeling.prony(xn, p = 1, q = 1)
    logger.info(res)


def test_shanks():
    N = 21
    T = 10 * (N - 1) + 1
    xn = np.ones(T)
    xn[N:] = 0

    expected_a = [1, -0.95]
    expected_b = [1, 0.301]
    expected_err = 3.95

    a, b, err = modeling.shanks(xn, p = 1, q = 1)

    assert np.array_equal(np.around(a, decimals=3), expected_a)
    assert np.array_equal(np.around(b, decimals=3), expected_b)
    assert round(err, 3) == expected_err


def test_spike():
    gn = np.array([-0.2, 0, 1])
    h, err = modeling.spike(gn, 4, 11)
    d = np.convolve(h, gn)
    logger.info(f"{h=}")
    logger.info(f"{err=}")
    logger.info(f"{d=}, {np.argmax(d)=}")


def test_ipf(): ...


def test_acm():
    x = np.ones(20)
    x[1::2] = x[1::2] * -1
    logger.info(x)

    a, err = modeling.acm(x, 2)
    logger.info(f"{a=}")
    logger.info(f"{err=}")


def test_covm():
    x = np.ones(20)
    x[1::2] = x[1::2] * -1
    logger.info(x)

    a, err = modeling.covm(x, 1)
    logger.info(f"{a=}")
    logger.info(f"{err=}")


def test_durbin():
    N = 64
    ap = [1, 0.7348, 1.882, 0.7057, 0.8851]

    zeros, poles, _ = signal.tf2zpk([1], ap)
    px = system.ARMA(p=poles, q=[1], N=64)
    logger.info(f"{px=}")

    rx = np.correlate(px, px, "same")
    logger.info(modeling.durbin(rx, p=4, q=0))
