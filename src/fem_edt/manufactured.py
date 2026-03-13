import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

Array = np.ndarray


def expm_via_eig(t: float, A: Array) -> Array:
    vals, vecs = np.linalg.eig(A)
    Vinv = np.linalg.inv(vecs)
    return (vecs @ np.diag(np.exp(t * vals)) @ Vinv).real


def N_quadratic(u: Array) -> Array:
    return np.array([u[0] ** 2, u[1] ** 2], dtype=float)


def N_sine(u: Array) -> Array:
    return np.sin(u)


# Experiment with different solution
def get_exact_solution(kind: str):
    if kind == "mixed_decay":

        def u_exact(t):
            return np.array(
                [np.exp(-t) * np.cos(2 * t), np.exp(-5 * t) * np.cos(2 * t)]
            )

        def du_exact(t):
            return np.array(
                [
                    -np.exp(-t) * np.cos(2 * t)
                    - 2 * np.exp(-t) * np.sin(2 * t),
                    -5 * np.exp(-5 * t) * np.cos(2 * t)
                    - 2 * np.exp(-5 * t) * np.sin(2 * t),
                ]
            )

    elif kind == "oscillatory":
        # Strong time variation -> harder for ETD1
        def u_exact(t):
            return np.array([np.exp(-t) * np.cos(3 * t), np.exp(-t) * np.sin(2 * t)])

        def du_exact(t):
            return np.array(
                [
                    -np.exp(-t) * np.cos(3 * t) - 3 * np.exp(-t) * np.sin(3 * t),
                    -np.exp(-t) * np.sin(2 * t) + 2 * np.exp(-t) * np.cos(2 * t),
                ]
            )

    elif kind == "stiffer_exact":
        # Faster decay modes -> more stiffness
        def u_exact(t):
            return np.array([np.exp(-10 * t), np.exp(-t)])

        def du_exact(t):
            return np.array(
                [-10 * np.exp(-10 * t), -np.exp(-t)]
            )

    elif kind == "pure_trig":
        # Pure trigonometric
        def u_exact(t):
            return np.array([np.sin(10*t), np.cos(10*t)])

        def du_exact(t):
            return np.array(
                [10*np.cos(10*t) , - 10*np.sin(10*t)]
            )

    return u_exact, du_exact


@dataclass(frozen=True)
class ManufacturedProblem:
    A: Array
    u_exact: Callable[[float], Array]
    du_exact: Callable[[float], Array]
    N: Callable[[Array], Array]
    b: Callable[[float, Array], Array]
    g: Callable[[float], Array]
    u0: Array


def make_linear_problem(A: Array, u0: Optional[Array] = None) -> ManufacturedProblem:
    """
        u'(t) = A u(t)
    Exact solution:
        u_exact(t) = exp(tA) u0
    """
    A = np.array(A, dtype=float)
    if u0 is None:
        u0 = np.ones(A.shape[0], dtype=float)

    def u_exact(t: float) -> Array:
        return expm_via_eig(t, A) @ u0

    def du_exact(t: float) -> Array:
        return A @ u_exact(t)

    def zero_array(t: Optional[float] = None, u: Array = None) -> Array:
        return np.zeros_like(u, dtype=float)

    return ManufacturedProblem(
        A=A,
        u_exact=u_exact,
        du_exact=du_exact,
        N=zero_array,
        b=zero_array,
        g=zero_array,
        u0=u0,
    )


def make_semilinear_problem(
    A: Array,
    u0: Optional[Array] = None,
    beta: float = 1,  # Size of nonlinear term.
    nonlinearity: str = "quadratic",
    kind: str = "oscillatory",
) -> ManufacturedProblem:
    """
    Manufactured semilinear problem

        u'(t) = A u(t) + g(t) + N(u(t))

    Choose exact solution u_exact(t), then define

        g(t) = u_exact'(t) - A u_exact(t) - N(u_exact(t))

    so that u_exact satisfies the equation exactly.
    """

    A = np.array(A, dtype=float)
    if u0 is None:
        u0 = np.ones(A.shape[0], dtype=float)

    if nonlinearity.lower() == "sine":
        N_base = N_sine
    else:
        N_base = N_quadratic

    def N(u: Array) -> Array:
        return beta * N_base(u)

    # Pick exact solution
    u_exact, du_exact = get_exact_solution(kind=kind)

    def g(t: float) -> Array:
        uex = u_exact(t)
        return du_exact(t) - A @ uex - N(uex)

    def b(t: float, u: Array) -> Array:
        return g(t) + N(u)

    return ManufacturedProblem(
        A=A,
        u_exact=u_exact,
        du_exact=du_exact,
        N=N,
        b=b,
        g=g,
        u0=u_exact(0.0),
    )
