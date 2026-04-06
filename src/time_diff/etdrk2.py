from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from .phi import PhiCache

Array = np.ndarray
BFunc = Callable[[float, Array], Array]


@dataclass
class SolverResult:
    t: Array
    u: Array
    h: float
    n_steps: int


def _as_vec(u: Array) -> Array:
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("State u must be a 1D array of shape (d,).")
    return u


def etdrk2_step(
    u: Array, t: float, h: float, A: Array, b: BFunc, cache: PhiCache
) -> Array:
    u = _as_vec(u)
    mats = cache.get([0, 1, 2])
    E = mats[0]
    phi1 = mats[1]
    phi2 = mats[2]

    U1 = u
    k1 = b(t, U1)
    U2 = (E @ u) + (h * (phi1 @ k1))
    k2 = b(t + h, U2)

    return (E @ u) + h * ((phi1 @ k1) + (phi2 @ (k2-k1)))


def etdrk2_solve(
    u0: Array,
    t0: float,
    T: float,
    h: float,
    A: Array,
    b: BFunc,
    cache: Optional[PhiCache] = None,
) -> SolverResult:

    u0 = _as_vec(u0)
    A = np.asarray(A, dtype=float)

    if cache is None:
        cache = PhiCache(A=A, h=h)

    times = [float(t0)]
    us = [u0.copy()]

    t = float(t0)
    u = u0.copy()

    while t < T - 1e-15:
        h_step = min(h, T - t)
        if abs(h_step - h) > 0:
            local_cache = PhiCache(A=A, h=h_step)
        else:
            local_cache = cache

        u = etdrk2_step(u, t, h_step, A, b, local_cache)
        t = t + h_step

        times.append(float(t))
        us.append(u.copy())

    t_arr = np.array(times, dtype=float)
    u_arr = np.vstack(us)
    return SolverResult(t=t_arr, u=u_arr, h=h, n_steps=len(times) - 1)
