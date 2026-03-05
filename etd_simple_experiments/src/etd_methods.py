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


def etd1_step(
    u: Array, t: float, h: float, A: Array, b: BFunc, cache: PhiCache
) -> Array:
    u = _as_vec(u)
    mats = cache.get([0, 1])
    E = mats[0]
    phi1 = mats[1]
    return (E @ u) + (h * (phi1 @ b(t, u)))


def etd1_solve(
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

    # Build time grid with last step adjusted if T-t0 not multiple of h
    times = [float(t0)]
    us = [u0.copy()]

    t = float(t0)
    u = u0.copy()

    # Use fixed-step loop; if final step is shorter, create a temporary cache
    while t < T - 1e-15:
        h_step = min(h, T - t)
        if abs(h_step - h) > 0:
            local_cache = PhiCache(A=A, h=h_step)
        else:
            local_cache = cache

        u = etd1_step(u, t, h_step, A, b, local_cache)
        t = t + h_step

        times.append(float(t))
        us.append(u.copy())

    t_arr = np.array(times, dtype=float)
    u_arr = np.vstack(us)
    return SolverResult(t=t_arr, u=u_arr, h=h, n_steps=len(times) - 1)


def etdrk2_step(
    u: Array, t: float, h: float, A: Array, b: BFunc, cache: PhiCache
) -> Array:
    u = _as_vec(u)
    mats = cache.get([0, 1, 2])
    E = mats[0]
    phi1 = mats[1]
    phi2 = mats[2]

    bn = b(t, u)
    a = (E @ u) + (h * (phi1 @ bn))
    bnp1 = b(t + h, a)

    corr = bnp1 - bn
    return (E @ u) + h * ((phi1 @ bn) + (phi2 @ corr))


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
