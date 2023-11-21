"""Systems of state matrix representation from the Appendix."""


import numpy.typing as npt


def convm(x: npt.ArrayLike, p: int = ...): ...

def covar(x: npt.ArrayLike, p: int = ...): ...

def normalprony(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()

def ywe(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()

def nywe(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()

def mywe(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()

def eywe(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()

def normaldeterministic(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()

def wienerhopf(x: npt.ArrayLike, p: int = ..., q: int = ...):
    raise NotImplementedError()
