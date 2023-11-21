"""Optimal Filters: optimal wiener filters optimal kalman estimator."""


"""Optimal kalman estimator"""
def kalman(y, A, C, sigmaw, sigmav): ...

"""Denoising based on IIR wiener filters."""
def wiener_denoise():
    raise NotImplementedError()

"""Systemid based on FIR wiener filters."""
def wiener_systemid():
    raise NotImplementedError()
