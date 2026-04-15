from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple
import numpy as np
from .phi import PhiCache, _phi_matrix_via_eig

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


def arnoldi(
    A: Callable[[Array], Array],
    v: Array,
    m: int,
) -> Tuple[Array, Array]:

    d = v.shape[0]
    V = np.zeros((d, m + 1), dtype=float)
    H = np.zeros((m + 1, m), dtype=float)

    beta = np.linalg.norm(v)
    if beta == 0.0:
        return V, H

    V[:, 0] = v / beta

    for j in range(m):
        w = A @ V[:, j]

        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            m = j + 1
            V = V[:, : m + 1]
            H = H[: m + 1, :m]
            break

        V[:, j + 1] = w / H[j + 1, j]

    return V, H


def phi_k_action_krylov(
    A: Callable[[Array], Array],
    v: Array,
    h: float,
    k: int,
    m: int = 30,
) -> Array:

    beta = np.linalg.norm(v)
    if beta == 0.0:
        return np.zeros_like(v)

    # Clamp m to at most d (cannot exceed problem size)
    d = v.shape[0]
    m = min(m, d)

    V, H = arnoldi(A, v, m)

    # Actual subspace size after possible early termination
    m_actual = H.shape[1]
    V_m = V[:, :m_actual]  # shape (d, m_actual)
    H_m = H[:m_actual, :m_actual]  # shape (m_actual, m_actual)

    # Compute phi_k(h * H_m) for the small Hessenberg matrix
    phi_Hm = _phi_matrix_via_eig(H_m, h, [k])[k]  # shape (m_actual, m_actual)

    e1 = np.zeros(m_actual)
    e1[0] = 1.0

    return beta * (V_m @ (phi_Hm @ e1))


def phi_actions_krylov(
    A: Callable[[Array], Array],
    v: Array,
    h: float,
    ks: Iterable[int],
    m: int = 30,
) -> Dict[int, Array]:

    ks = sorted(set(int(k) for k in ks))
    beta = np.linalg.norm(v)

    if beta == 0.0:
        return {k: np.zeros_like(v) for k in ks}

    d = v.shape[0]
    m = min(m, d)

    V, H = arnoldi(A, v, m)

    m_actual = H.shape[1]
    V_m = V[:, :m_actual]
    H_m = H[:m_actual, :m_actual]

    phi_mats = _phi_matrix_via_eig(H_m, h, ks)

    e1 = np.zeros(m_actual)
    e1[0] = 1.0

    return {k: beta * (V_m @ (phi_mats[k] @ e1)) for k in ks}


def etd1_step_krylov(
    u: Array,
    t: float,
    h: float,
    A: Callable[[Array], Array],
    b: Callable[[float, Array], Array],
    m: int = 30,
) -> Array:

    u = np.asarray(u, dtype=float)
    bu = b(t, u)

    # phi_0(hA) @ u  and  phi_1(hA) @ b(t, u) from a single Krylov basis each.
    phi0_u = phi_k_action_krylov(A, u, h, k=0, m=m)
    phi1_bu = phi_k_action_krylov(A, bu, h, k=1, m=m)

    return phi0_u + h * phi1_bu


def etd1_solve_krylov(
    u0: Array,
    t0: float,
    T: float,
    h: float,
    A: Array,
    b: BFunc,
    m: int,
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

        u = etd1_step_krylov(u=u, t=t, h=h_step, A=A, b=b, m=m)
        t = t + h_step

        times.append(float(t))
        us.append(u.copy())

    t_arr = np.array(times, dtype=float)
    u_arr = np.vstack(us)
    return SolverResult(t=t_arr, u=u_arr, h=h, n_steps=len(times) - 1)
