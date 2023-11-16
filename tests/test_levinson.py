""""""


import numpy as np
from pyssp.levinson import glev, gtor


def test_glev():
    '''Example 5.3.1, Page 266'''
    r = [4, 2, 1]
    b = [9, 6, 12]

    res = glev(r, b)
    print(res)


def test_gtor():
    '''Based on example 5.2.6'''

    gamma = [1/2, 1/2, 1/2]
    epsilon = 2 * (3 / 4)**3
    res = gtor(gamma, epsilon)
    true_results = np.array([2, -1, -1/4, 1/8])
    print(res)