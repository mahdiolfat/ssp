"""Implementation of algorithm from Chapter 6."""


import numpy as np
import scipy as sp


def fcov(x, p):
    '''
    Figure 6.15, Page 310.

    Using the forward covariance method the reflection co-efficients of the lattice filter
    are found by sequentially minimizing the sum of the squares of the forward prediction error.
    '''

    if p >= len(x):
        raise ValueError("Model order must be less than length of signal")

    _x = np.array(x).reshape(-1, 1)
    N = len(x)
    eplus = _x[1:N]
    eminus = _x[:N - 1]

    gamma = np.empty((p, 1))
    err = np.empty((p, 1))

    for j in range(p):
        print(j)
        N = N - 1
        # print(f"{eplus=}, {eplus.shape=}")
        # print(f"{eminus=}, {eminus.shape=}")
        gamma[j] = (np.transpose(-eminus) @ eplus) / (np.transpose(eminus) @ eminus)
        temp1 = eplus + gamma[j] * eminus
        temp2 = eminus + np.conjugate(gamma[j]) * eplus
        err[j] = np.transpose(temp1) @ temp1
        eplus = temp1[1:N]
        eminus = temp2[:N - 1]
        print(gamma)
        print(err)
        print()

    return gamma, err


def burg(x, p):
    '''
    Sequentially minimizes the sum of the forward and backward covariance errors.

    Guaranteed to be stable. All reflection coefficients will be <|1|
    '''

    if p > len(x):
        raise ValueError("Model order must be less than length of signal")

    _x = np.array(x).reshape(-1, 1)
    N = len(x)
    eplus = _x[1:N]
    eminus = _x[:N - 1]

    gamma = np.empty((p, 1))
    err = np.empty((p, 1))

    for j in range(p):
        print(j)
        N = N - 1
        # print(f"{eplus=}, {eplus.shape=}")
        # print(f"{eminus=}, {eminus.shape=}")
        eplusmag = np.transpose(eplus) @ eplus
        eminusmag = np.transpose(eplus) @ eplus
        gamma[j] = (np.transpose(-2 * eminus) @ eplus) / (eplusmag + eminusmag)
        temp1 = eplus + gamma[j] * eminus
        temp2 = eminus + np.conjugate(gamma[j]) * eplus
        err[j] = np.transpose(temp1) @ temp1 + np.transpose(temp2) @ temp2
        eplus = temp1[1:N]
        eminus = temp2[:N - 1]
        print()

    return gamma, err


def bcov():
    '''
    Sequentially minimizes the backward covariance error.

    Arguements: (x, p)
    '''


def mcov(x, p):
    '''
    Modified covariance method. Unlike the forward/backward algorithms,
    it *does not* minimize an error term sequentially.
    '''
    _x = np.array(x).reshape(-1, 1)
    N = len(x)

    if p >= len(x):
        raise ValueError("Model order must be less than length of signal")

    X = sp.linalg.toeplitz(_x[p:N], np.flipud(x[:p + 1]))
    R = np.transpose(X) @ X
    R1 = np.array(R[1:p + 1, 1: p + 1])
    R2 = np.array(np.flipud(np.fliplr(R[:p, :p])))
    b1 = np.array(R[1:p + 1, 1])
    b2 = np.array(np.flipud(R[:p, p]))

    Rx = -R1 - R2
    b = b1 + b2
    a = sp.linalg.solve_toeplitz(Rx[:, 1], b)
    a = np.concatenate(([1], a))
    print(a.shape)
    err = np.dot(R[0], a) + np.dot(np.flip(R[p]), a)

    return a, err
