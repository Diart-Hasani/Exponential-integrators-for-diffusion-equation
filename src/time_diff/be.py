from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

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


def be_step(
    u: Array,
    h: float,
    A: Array,
) -> Array:
    """
    One backward Euler step for linear system.
    """
    u = _as_vec(u)
    A = np.asarray(A, dtype=float)

    d = u.size
    M = np.eye(d) - h * A
    sol = np.linalg.solve(M, u)

    return sol


def be_step_newton(
    u: Array,
    t: float,
    h: float,
    A: Array,
    b: BFunc,
    fp_iters: int = 20,
    tol: float = 1e-12,
) -> Array:
    """
    One backward Euler step for with Newton iterations
    for nonlinear system.
    """
    u = _as_vec(u)
    A = np.asarray(A, dtype=float)

    d = u.size
    t_next = t + h
    M = np.eye(d) - h * A

    # Newton method
    u_guess = u.copy()
    for _ in range(fp_iters):
        rhs = u + h * b(t_next, u_guess)
        u_new = np.linalg.solve(M, rhs)

        if np.linalg.norm(u_new - u_guess, ord=np.inf) < tol:
            return u_new
        u_guess = u_new

    return u_guess


def backward_euler_solve(
    u0: Array,
    t0: float,
    T: float,
    h: float,
    A: Array,
    b: Optional[BFunc] = None,
    fp_iters: int = 20,
    tol: float = 1e-12,
) -> SolverResult:
    u0 = _as_vec(u0)
    A = np.asarray(A, dtype=float)

    if b is None:
        def b(t: float, u: Array) -> Array:
            return np.zeros_like(u, dtype=float)

    times = [float(t0)]
    us = [u0.copy()]

    t = float(t0)
    u = u0.copy()

    while t < T - 1e-15:
        h_step = min(h, T - t)

        if not np.any(b):
            u = be_step(u, h, A)
        else:
            u = be_step_newton(u, t, h_step, A, b, fp_iters=fp_iters, tol=tol)
        t = t + h_step

        times.append(float(t))
        us.append(u.copy())

    t_arr = np.array(times, dtype=float)
    u_arr = np.vstack(us)
    return SolverResult(t=t_arr, u=u_arr, h=h, n_steps=len(times) - 1)
