from __future__ import annotations

import os
import numpy as np
import time

from src.time_diff.etd1 import etd1_solve
from src.time_diff.be import backward_euler_solve
from src.time_diff.krylov_method import etd1_solve_krylov
from src.fem.mesh_1d import interval_mesh, boundary_nodes
from src.fem.fem1d_assembly import assemble_matrices


def u0_fun(x: np.ndarray) -> np.ndarray:
    """Initial condition u0(x) = x(1-x)."""
    return x * (1.0 - x)


def fourier_exact(
    x: np.ndarray,
    t: float,
    kappa: float = 1.0,
    n_modes: int = 400,
) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)

    for n in range(1, n_modes + 1, 2):
        a_n = 8.0 / ((n * np.pi) ** 3)
        u += a_n * np.exp(-kappa * (n * np.pi) ** 2 * t) * np.sin(n * np.pi * x)

    return u


def find_optimal_modes(
    kappa: float = 1.0,
    L: float = 1.0,
    tol: float = 1e-10,
    max_modes: int = 10_000,
) -> int:
    """
    Find the smallest odd K such that adding the next odd term changes
    the L2 norm of the Fourier series by less than tol.
    """
    # Use t=0 as the default: no exponential decay, so convergence is slowest
    t = 0.0

    phi_norm = np.sqrt(L / 2.0)

    K = 1
    while K + 2 <= max_modes:
        k_next = K + 2
        lam = (k_next * np.pi / L) ** 2
        c_k = 8.0 * L**2 / (k_next * np.pi) ** 3
        decay = np.exp(-kappa * lam * t)

        increment = abs(c_k) * decay * phi_norm

        if increment < tol:
            print(
                f"K = {K:>6} odd modes  "
                f"(last increment = {increment:.2e} < tol = {tol:.2e})"
            )
            return K

        K = k_next

    print(f"Warning: max_modes={max_modes} reached without convergence.")
    return K


def build_reduced_system_1d(n_elements: int, lx: float = 1.0):
    """
    Build a 1D mesh, assemble full FEM matrices, and restrict to interior nodes.
    """
    mesh = interval_mesh(nx=n_elements, lx=lx)
    M, K = assemble_matrices(mesh)

    all_nodes = np.arange(mesh.nodes.shape[0])
    bd = boundary_nodes(mesh)
    # Nodes that are different from all_nodes and bd
    interior = np.setdiff1d(all_nodes, bd)

    Mii = M[interior][:, interior].toarray()
    Kii = K[interior][:, interior].toarray()
    x_full = mesh.nodes[:, 0]
    x_int = x_full[interior]

    return mesh, interior, bd, Mii, Kii, x_full, x_int


def fem_initial_vector(x_int: np.ndarray) -> np.ndarray:
    """Nodal interpolation of the initial condition."""
    return u0_fun(x_int)


def add_boundaries(u_int: np.ndarray) -> np.ndarray:
    """Add homogeneous Dirichlet values at x=0 and x=1."""
    return np.concatenate(([0.0], u_int, [0.0]))


def l2_error_on_nodes(x: np.ndarray, e: np.ndarray) -> float:
    """Nodal L2 error computed by trapezoidal rule."""
    return np.sqrt(np.trapezoid(e**2, x))


def solve_fem_etd1(
    n_elements: int,
    lx: float,
    dt: float,
    T: float,
    kappa: float = 1.0,
):
    """
    Solve

        M U'(t) + kappa K U(t) =

    on the interior nodes using etd1.
    """
    mesh, interior, bd, Mii, Kii, x_full, x_int = build_reduced_system_1d(
        n_elements=n_elements,
        lx=lx,
    )

    A = -np.linalg.solve(Mii, kappa * Kii)
    u0 = fem_initial_vector(x_int)

    def b(t: float, u: np.ndarray, L: float = 1.0) -> np.ndarray:
        return np.zeros_like(u)

    sol = etd1_solve(
        u0=u0,
        t0=0.0,
        T=T,
        h=dt,
        A=A,
        b=b,
    )

    return mesh, interior, x_full, x_int, sol


def solve_fem_be(
    n_elements: int,
    lx: float,
    dt: float,
    T: float,
    kappa: float = 1.0,
):
    """
    Solve

        M U'(t) + kappa K U(t) = 0

    on the interior nodes using etd1.
    """
    mesh, interior, bd, Mii, Kii, x_full, x_int = build_reduced_system_1d(
        n_elements=n_elements,
        lx=lx,
    )

    A = -np.linalg.solve(Mii, kappa * Kii)
    u0 = fem_initial_vector(x_int)

    def b(t: float, u: np.ndarray, L: float = 1.0) -> np.ndarray:
        return np.zeros_like(u)

    sol = backward_euler_solve(
        u0=u0,
        t0=0.0,
        T=T,
        h=dt,
        A=A,
        b=b,
    )

    return mesh, interior, x_full, x_int, sol


def solve_fem_etd1_krylov(
    n_elements: int,
    lx: float,
    dt: float,
    T: float,
    kappa: float = 1.0,
):
    """
    Solve

        M U'(t) + kappa K U(t) = 0

    on the interior nodes using etd1.
    """
    mesh, interior, bd, Mii, Kii, x_full, x_int = build_reduced_system_1d(
        n_elements=n_elements,
        lx=lx,
    )

    A = -np.linalg.solve(Mii, kappa * Kii)
    u0 = fem_initial_vector(x_int)

    def b(t: float, u: np.ndarray, L: float = 1.0) -> np.ndarray:
        return np.zeros_like(u)

    sol = etd1_solve_krylov(
        u0=u0,
        t0=0.0,
        T=T,
        h=dt,
        A=A,
        b=b,
        m=10,
    )

    return mesh, interior, x_full, x_int, sol


# Calculate error for each delta t
def error_calc(
    method: str = "etd1",
    kappa: float = 1.0,
    T: float = 1.0,
    dt: float = 0.001,
    lx: float = 1.0,
    n_elements: int = 200,
    n_modes: int = 500,
    output_dir: str = None,
) -> None:

    err_inf = 0
    err_l2 = 0

    start = time.perf_counter()
    if method == "etd1":
        mesh, interior, x_full, x_int, sol = solve_fem_etd1(
            n_elements=n_elements,
            lx=lx,
            dt=dt,
            T=T,
            kappa=kappa,
        )
    elif method == "be":
        mesh, interior, x_full, x_int, sol = solve_fem_be(
            n_elements=n_elements,
            lx=lx,
            dt=dt,
            T=T,
            kappa=kappa,
        )
    elif method == "etd1_krylov":
        mesh, interior, x_full, x_int, sol = solve_fem_etd1_krylov(
            n_elements=n_elements,
            lx=lx,
            dt=dt,
            T=T,
            kappa=kappa,
        )
    elapsed = time.perf_counter() - start

    u_num = add_boundaries(sol.u[-1])
    u_ex = fourier_exact(x_full, T, kappa=kappa, n_modes=n_modes)
    e = u_num - u_ex

    err_inf = np.max(np.abs(e))
    err_l2 = l2_error_on_nodes(x_full, e)

    print(f"Max error using {method}: ", err_inf)
    print(f"L2 error using {method}: ", err_l2)
    print("Time of calculation: ", elapsed)


def main():
    output_dir = "results/ibvp_etd"
    os.makedirs(output_dir, exist_ok=True)

    kappa = 2.0
    T = 0.2
    dt = 0.001
    lx = 1.0
    n_elements = 1000

    n_optimal = find_optimal_modes(kappa=kappa, L=1, tol=1e-9)

    error_calc(
        n_elements=n_elements,
        method="etd1",
        kappa=kappa,
        T=T,
        dt=dt,
        lx=lx,
        output_dir=output_dir,
        n_modes=n_optimal,
    )

    error_calc(
        n_elements=n_elements,
        method="etd1_krylov",
        kappa=kappa,
        T=T,
        dt=dt,
        lx=lx,
        output_dir=output_dir,
        n_modes=n_optimal,
    )
    """
    error_calc(
        n_elements=n_elements,
        method="etd1_krylov",
        kappa=kappa,
        T=T,
        dt=dt,
        lx=lx,
        output_dir=output_dir,
        n_modes=n_optimal,
    )
    """

if __name__ == "__main__":
    main()
