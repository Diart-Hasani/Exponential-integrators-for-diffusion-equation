import numpy as np
from dataclasses import dataclass
from typing import Callable


Array = np.ndarray


def u_exact(t: float) -> Array:
    """Exact solution u(t) = [cos t, sin t]."""
    return np.array([np.cos(t), np.sin(t)], dtype=float)


def du_exact(t: float) -> Array:
    """Derivative u'(t) = [-sin t, cos t]."""
    return np.array([-np.sin(t), np.cos(t)], dtype=float)


def N_square(u: Array) -> Array:
    """
    Nonlinearity N(u) = [u1^2, u2^2].
    (Componentwise square, genuinely nonlinear.)
    """
    return np.array([u[0] ** 2, u[1] ** 2], dtype=float)


@dataclass(frozen=True)
class ManufacturedProblem:
    """
    Container for a manufactured semilinear problem:
        u'(t) = A u(t) + b(t, u(t))
    with known exact solution u_exact(t).
    """
    A: Array
    u_exact: Callable[[float], Array]
    du_exact: Callable[[float], Array]
    N: Callable[[Array], Array]
    b: Callable[[float, Array], Array]
    g: Callable[[float], Array]
    u0: Array


def make_problem_cos_sin_with_square_nonlinearity(A: Array) -> ManufacturedProblem:
    """
    Builds b(t,u) = g(t) + N(u) such that u_exact(t) = [cos t, sin t]
    is the exact solution of:
        u' = A u + b(t,u)
    with N(u) = [u1^2, u2^2].

    g(t) is defined by:
        g(t) = u'(t) - A u(t) - N(u(t))
    """
    def g(t: float) -> Array:
        u = u_exact(t)
        return du_exact(t) - (A @ u) - N_square(u)

    def b(t: float, u: Array) -> Array:
        return g(t) + N_square(u)

    u0 = u_exact(0.0)

    return ManufacturedProblem(
        A=A,
        u_exact=u_exact,
        du_exact=du_exact,
        N=N_square,
        b=b,
        g=g,
        u0=u0,
    )