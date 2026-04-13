from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.time_diff.etd1 import etd1_solve
from src.time_diff.etdrk2 import etdrk2_solve
from src.fem.mesh_1d import interval_mesh, boundary_nodes
from src.fem.fem1d_assembly import assemble_matrices


def u0_fun(x: np.ndarray) -> np.ndarray:
    """Initial condition u0(x) = x(1-x)."""
    return x * (1.0 - x)


def fourier_exact(
    x: np.ndarray,
    t: float,
    kappa: float = 1.0,
    L: float = 1.0,
    m: int = 5,  # which sine mode is forced
    forcing: str = "cos",  # "cos" or "linear"
    n_modes: int = 400,
) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)

    for k in range(1, n_modes + 1, 2):  # only odd k have nonzero c_k
        mu_k = kappa * (k * np.pi / L) ** 2
        c_k = 8.0 * L**2 / (k * np.pi) ** 3

        # Homogeneous part: same for all modes
        u_hat_k = c_k * np.exp(-mu_k * t)

        # Particular part: only for the forced mode k == m
        if k == m:
            mu_m = mu_k
            if forcing == "cos":
                I_m = (mu_m * (np.cos(t) - np.exp(-mu_m * t)) + np.sin(t)) / (
                    mu_m**2 + 1.0
                )
            elif forcing == "linear":
                I_m = (3.0 / mu_m) * (t - (1.0 - np.exp(-mu_m * t)) / mu_m)
            else:
                raise ValueError(f"Unknown forcing '{forcing}'")

            u_hat_k += I_m

        u += u_hat_k * np.sin(k * np.pi * x / L)

    return u


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
        lx=1.0,
    )

    A = -np.linalg.solve(Mii, kappa * Kii)
    u0 = fem_initial_vector(x_int)

    def b(t: float, u: np.ndarray, L: float = 1.0) -> np.ndarray:
        # f(t) * sin(m*pi*x/L) projected onto interior nodes
        # since we use nodal interpolation, this is just pointwise evaluation
        f_t = np.cos(t)  # or 3*t
        m = 5
        return f_t * np.sin(m * np.pi * x_int / L)

    sol = etd1_solve(
        u0=u0,
        t0=0.0,
        T=T,
        h=dt,
        A=A,
        b=b,
    )

    return mesh, interior, x_full, x_int, sol


def solve_fem_etdrk2(
    n_elements: int,
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
        lx=1.0,
    )

    A = -np.linalg.solve(Mii, kappa * Kii)
    u0 = fem_initial_vector(x_int)

    def b(t: float, u: np.ndarray, L: float = 1.0) -> np.ndarray:
        # f(t) * sin(m*pi*x/L) projected onto interior nodes
        # since we use nodal interpolation, this is just pointwise evaluation
        f_t = np.cos(t)  # or 3*t
        m = 5
        return f_t * np.sin(m * np.pi * x_int / L)

    sol = etdrk2_solve(
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

    (line_num,) = ax.plot([], [], "--", linewidth=2, label="FEM + etd1")
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


# Calculate error for each size of elements and delta t
def error_calc(
    method: str = "", kappa: float = 1.0, T: float = 1.0, output_dir: str = None
) -> None:
    n_elements_list = [20, 40, 80, 160]
    dt_list = [0.1, 0.05, 0.025, 0.0125]

    err_inf = np.zeros((len(n_elements_list), len(dt_list)))
    err_l2 = np.zeros((len(n_elements_list), len(dt_list)))

    for i, n_el in enumerate(n_elements_list):
        for j, dt in enumerate(dt_list):
            if method == "etd1":
                mesh, interior, x_full, x_int, sol = solve_fem_etd1(
                    n_elements=n_el,
                    dt=dt,
                    T=T,
                    kappa=kappa,
                )
            elif method == "etdrk2":
                mesh, interior, x_full, x_int, sol = solve_fem_etdrk2(
                    n_elements=n_el,
                    dt=dt,
                    T=T,
                    kappa=kappa,
                )

            u_num = add_boundaries(sol.u[-1])
            u_ex = fourier_exact(x_full, T, kappa=kappa, n_modes=800)
            e = u_num - u_ex

            err_inf[i, j] = np.max(np.abs(e))
            err_l2[i, j] = l2_error_on_nodes(x_full, e)

    # Plot error sclaing with mesh scaling
    fixed_dt_index = -1
    fixed_dt = dt_list[fixed_dt_index]
    h_list = np.array([1.0 / n for n in n_elements_list])

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.loglog(
        h_list, err_inf[:, fixed_dt_index], "o-", label=f"max error, dt={fixed_dt}"
    )
    ax2.loglog(
        h_list, err_l2[:, fixed_dt_index], "s-", label=f"L2 error, dt={fixed_dt}"
    )

    ref = err_l2[-1, fixed_dt_index] * (h_list / h_list[-1]) ** 2
    ax2.loglog(h_list, ref, "k--", label="reference slope 2")

    ax2.set_xlabel("mesh size h")
    ax2.set_ylabel("error at final time")
    ax2.set_title(f"Error of {method} vs mesh size")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"mesh_error_{method}.png"), dpi=300)
    plt.close(fig2)

    # Plot error scaling with delta time scaling
    fixed_mesh_index = 0
    fixed_n_el = n_elements_list[fixed_mesh_index]

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.loglog(
        dt_list,
        err_inf[fixed_mesh_index, :],
        "o-",
        label=f"max error, n_el={fixed_n_el}",
    )
    ax3.loglog(
        dt_list, err_l2[fixed_mesh_index, :], "s-", label=f"L2 error, n_el={fixed_n_el}"
    )

    ax3.set_xlabel("time step dt")
    ax3.set_ylabel("error at final time")
    ax3.set_title(f"Error of {method} vs time step")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, f"dt_error_{method}.png"), dpi=300)
    plt.close(fig3)


def run_experiment():
    output_dir = "results/ibvp_fourier_etd"
    os.makedirs(output_dir, exist_ok=True)

    kappa = 1.5
    T = 0.2
    dt = 0.1

    # method: "etd1" or "etdrk2"
    error_calc(method="etd1", kappa=kappa, T=T, output_dir=output_dir)
    error_calc(method="etdrk2", kappa=kappa, T=T, output_dir=output_dir)

    # Make an animation for etd1
    mesh, interior, x_full, x_int, sol = solve_fem_etd1(
        n_elements=50,
        dt=dt,
        T=T,
        kappa=kappa,
    )

    u_num_snapshots = np.array([add_boundaries(u_int) for u_int in sol.u])
    u_exact_snapshots = np.array(
        [fourier_exact(x_full, t, kappa=kappa, n_modes=1000) for t in sol.t]
    )

    animate_solution_1d(
        x=x_full,
        u_snapshots=u_num_snapshots,
        times=sol.t,
        save_path=os.path.join(output_dir, "solution_animation_etd1.gif"),
        u_exact_snapshots=u_exact_snapshots,
    )

    # Make an animation for etdrk2
    mesh, interior, x_full, x_int, sol = solve_fem_etdrk2(
        n_elements=50,
        dt=dt,
        T=T,
        kappa=kappa,
    )

    u_num_snapshots = np.array([add_boundaries(u_int) for u_int in sol.u])
    u_exact_snapshots = np.array(
        [fourier_exact(x_full, t, kappa=kappa, n_modes=1000) for t in sol.t]
    )

    animate_solution_1d(
        x=x_full,
        u_snapshots=u_num_snapshots,
        times=sol.t,
        save_path=os.path.join(output_dir, "solution_animation_etdrk2.gif"),
        u_exact_snapshots=u_exact_snapshots,
    )

    # Plot the function for different times
    n_el_plot = 100
    dt_plot = 0.01
    times_to_plot = [0.0, 0.02, 0.04, 0.08]

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    for t_plot in times_to_plot:
        mesh, interior, x_full, x_int, sol = solve_fem_etd1(
            n_elements=n_el_plot,
            dt=dt_plot,
            T=t_plot,
            kappa=kappa,
        )

        u_num = add_boundaries(sol.u[-1])
        u_ex = fourier_exact(x_full, t_plot, kappa=kappa, n_modes=800)

        ax1.plot(x_full, u_ex, "-", linewidth=2, label=f"Fourier, t={t_plot:g}")
        ax1.plot(x_full, u_num, "--", linewidth=1.5, label=f"FEM+etd1, t={t_plot:g}")

    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x,t)")
    ax1.set_title("1D heat equation: Fourier vs FEM+etd1")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "solution_compare.png"), dpi=300)
    plt.close(fig1)


if __name__ == "__main__":
    run_experiment()
