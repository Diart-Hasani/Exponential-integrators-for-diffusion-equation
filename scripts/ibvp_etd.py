from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.time_diff.etd1 import etd1_solve
from src.fem.mesh_1d import interval_mesh, boundary_nodes
from src.fem.fem1d_assembly import assemble_matrices
from src.time_diff.be import backward_euler_solve


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


def animate_solution_1d(
    x: np.ndarray,
    u_snapshots: np.ndarray,
    times: np.ndarray,
    method: str,
    save_path: str,
    u_exact_snapshots: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Automatic calculation of limits for y-axis
    ymin = np.min(u_snapshots)
    ymax = np.max(u_snapshots)

    if u_exact_snapshots is not None:
        ymin = min(ymin, np.min(u_exact_snapshots))
        ymax = max(ymax, np.max(u_exact_snapshots))

    pad = 0.05 * max(1.0, ymax - ymin)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.grid(True, alpha=0.3)

    (line_num,) = ax.plot([], [], "--", linewidth=2, label=f"FEM + {method}")
    if u_exact_snapshots is not None:
        (line_ex,) = ax.plot([], [], "-", linewidth=2, label="Fourier")
    else:
        line_ex = None

    ax.legend()

    def update(k: int):
        line_num.set_data(x, u_snapshots[k])

        artists = [line_num]
        if line_ex is not None:
            line_ex.set_data(x, u_exact_snapshots[k])
            artists.append(line_ex)

        ax.set_title(f"1D heat equation, t = {times[k]:.4f}")
        return tuple(artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=100,
        blit=False,
    )
    anim.save(save_path, writer=PillowWriter(fps=5))
    plt.close(fig)


# Calculate error for each delta t
def error_calc(
    method: str = "etd1",
    kappa: float = 1.0,
    T: float = 1.0,
    lx: float = 1.0,
    n_elements: int = 200,
    n_modes: int = 500,
    output_dir: str = None,
) -> None:

    dt_list = np.array([0.1, 0.01, 0.001])

    err_inf = np.zeros(len(dt_list))
    err_l2 = np.zeros(len(dt_list))

    for j, dt in enumerate(dt_list):
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

        u_num = add_boundaries(sol.u[-1])
        u_ex = fourier_exact(x_full, T, kappa=kappa, n_modes=n_modes)
        e = u_num - u_ex

        err_inf[j] = np.max(np.abs(e))
        err_l2[j] = l2_error_on_nodes(x_full, e)

    # Plot error scaling with delta time scaling
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.loglog(
        dt_list,
        err_inf,
        "o-",
        label="max error",
    )
    ax3.loglog(dt_list, err_l2, "s-", label="L2 error")
    ref = err_l2[0] * (dt_list / dt_list[0])
    ax3.loglog(dt_list, ref, "k--", label="reference slope 2")

    ax3.set_xlabel("time step dt")
    ax3.set_ylabel("error at final time")
    ax3.set_title(f"Error of {method} vs time step")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, f"dt_error_{method}.png"), dpi=300)
    plt.close(fig3)


def main():
    output_dir = "results/ibvp_etd"
    os.makedirs(output_dir, exist_ok=True)

    kappa = 2.0
    T = 0.2
    dt = 0.001
    lx = 1.0
    n_elements = 500

    n_optimal = find_optimal_modes(kappa=kappa, L=1, tol=1e-9)

    error_calc(
        n_elements=n_elements,
        method="be",
        kappa=kappa,
        T=T,
        output_dir=output_dir,
        n_modes=n_optimal,
    )
    error_calc(
        n_elements=n_elements,
        method="be",
        kappa=kappa,
        T=T,
        output_dir=output_dir,
        n_modes=n_optimal,
    )

    make_animation = False
    if make_animation:
        kappa = 2.0
        T = 0.2
        dt = 0.01
        lx = 1.0
        n_elements = 160

        # Make an animation for etd1
        mesh, interior, x_full, x_int, sol = solve_fem_etd1(
            n_elements=n_elements,
            lx=lx,
            dt=dt,
            T=T,
            kappa=kappa,
        )

        u_num_snapshots = np.array([add_boundaries(u_int) for u_int in sol.u])
        u_exact_snapshots = np.array(
            [fourier_exact(x_full, t, kappa=kappa, n_modes=n_optimal) for t in sol.t]
        )

        animate_solution_1d(
            x=x_full,
            u_snapshots=u_num_snapshots,
            times=sol.t,
            method="etd1",
            save_path=os.path.join(output_dir, "solution_animation_etd1.gif"),
            u_exact_snapshots=u_exact_snapshots,
        )

        # Make an animation for be
        mesh, interior, x_full, x_int, sol = solve_fem_be(
            n_elements=n_elements,
            lx=lx,
            dt=dt,
            T=T,
            kappa=kappa,
        )

        u_num_snapshots = np.array([add_boundaries(u_int) for u_int in sol.u])
        u_exact_snapshots = np.array(
            [fourier_exact(x_full, t, kappa=kappa, n_modes=n_optimal) for t in sol.t]
        )

        animate_solution_1d(
            x=x_full,
            u_snapshots=u_num_snapshots,
            times=sol.t,
            method="be",
            save_path=os.path.join(output_dir, "solution_animation_be.gif"),
            u_exact_snapshots=u_exact_snapshots,
        )

    make_time_plots = False
    if make_time_plots:
        # Plot the function for different times
        n_el_plot = 160
        lx = 1.0
        kappa = 2.0
        dt_plot = 0.01
        times_to_plot = [0.0, 0.02, 0.04, 0.08, 0.16]

        fig1, ax1 = plt.subplots(figsize=(8, 5))

        for t_plot in times_to_plot:
            mesh, interior, x_full, x_int, sol = solve_fem_be(
                n_elements=n_el_plot,
                lx=lx,
                dt=dt_plot,
                T=t_plot,
                kappa=kappa,
            )

            u_num = add_boundaries(sol.u[-1])
            u_ex = fourier_exact(x_full, t_plot, kappa=kappa, n_modes=n_optimal)

            ax1.plot(x_full, u_ex, "-", linewidth=2, label=f"Fourier, t={t_plot:g}")
            ax1.plot(x_full, u_num, "--", linewidth=1.5, label=f"FEM+be, t={t_plot:g}")

        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x,t)")
        ax1.set_title("1D heat equation: Fourier vs FEM+be")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, ncol=2)
        fig1.tight_layout()
        fig1.savefig(os.path.join(output_dir, "solution_compare.png"), dpi=300)
        plt.close(fig1)


if __name__ == "__main__":
    main()
