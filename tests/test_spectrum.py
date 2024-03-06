"""
Example 8.6.2: 8.31
Frequency Estimators: Example 8.6.5. Page 468.
"""

import logging

from typing import NoReturn
from ssp import spectrum


logger = logging.getLogger(__name__)


def test_phd() -> NoReturn:
    raise NotImplementedError

def test_music() -> NoReturn:
    raise NotImplementedError

# Frequency Estimators: Noise subspace methods
def test_ev() -> NoReturn:
    raise NotImplementedError

def test_min_norm() -> NoReturn:
    raise NotImplementedError

# Principle component methods, Spectrum
def test_bt_pc() -> NoReturn:
    raise NotImplementedError

def test_mv_pc() -> NoReturn:
    raise NotImplementedError

def test_ar_pc() -> NoReturn:
    raise NotImplementedError

def test_overlay() -> None:
    res = spectrum.overlay(64, [0.2, 0.9], [1, 2], 0.5, 10)

    assert(res.shape == (1024, 64))
