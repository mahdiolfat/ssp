"""
Example 8.6.2: 8.31
Frequency Estimators: Example 8.6.5. Page 468.
"""

import logging

from typing import NoReturn
from ssp import spectrum, state
import numpy as np


logger = logging.getLogger(__name__)


def test_phd() -> None:
    # two complex exponentials in white noise have the following autocorrelation sequence
    rx = np.array([6, 1.92705 + 4.58522j, -3.342705 + 3.49541j], dtype=complex)
    logger.warning(f'{rx=}')
    

#def test_music() -> NoReturn:
    #raise NotImplementedError

## Frequency Estimators: Noise subspace methods
#def test_ev() -> NoReturn:
    #raise NotImplementedError

#def test_min_norm() -> NoReturn:
    #raise NotImplementedError

## Principle component methods, Spectrum
#def test_bt_pc() -> NoReturn:
    #raise NotImplementedError

#def test_mv_pc() -> NoReturn:
    #raise NotImplementedError

#def test_ar_pc() -> NoReturn:
    #raise NotImplementedError

#def test_overlay() -> None:
    #res = spectrum.overlay(64, [0.2, 0.9], [1, 2], 0.5, 10)

    #assert(res.shape == (1024, 64))
