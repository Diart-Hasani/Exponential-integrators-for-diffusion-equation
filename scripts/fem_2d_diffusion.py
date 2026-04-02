import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colors

from time_diff.be import backward_euler_solve
from time_diff.etd1 import etd1_solve
from fem.mesh_2d import rectangle_mesh, plot_mesh, boundary_nodes
from fem.fem2d_assembly import assemble_matrices


def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Initial value v(x,y).
    """
    ini_con = np.sin(np.pi * x) * np.sin(np.pi * y)
    return ini_con


def source_term(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """
    Right-hand side f(x,y,t).
    """
    return np.zeros_like(x)


def dirichlet_value(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """
    Boundary data g(x,y,t) on Gamma.
    """
    return np.zeros_like(x)


def build_load_vector(mesh, t: float) -> np.ndarray:
    """
    Load vector:

    """
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    f_nodal = source_term(x, y, t)
    return f_nodal


def plot_solution_3d(mesh, u_h: np.ndarray, save_path: str, title: str = "FEM solution") -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        u_h,
        triangles=mesh.elements,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.2,
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u_h")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def animate_solution_3d(
    mesh,
    u_snapshots: list[np.ndarray],
    snapshots_times: np.ndarray,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    zmin = min(np.min(u) for u in u_snapshots)
    zmax = max(np.max(u) for u in u_snapshots)
    norm = colors.Normalize(vmin=zmin, vmax=zmax)

    def update(k):
        ax.clear()

        surf = ax.plot_trisurf(
            mesh.nodes[:, 0],
            mesh.nodes[:, 1],
            u_snapshots[k],
            triangles=mesh.elements,
            cmap="viridis",
            norm=norm,
            edgecolor="k",
            linewidth=0.2,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u_h")
        ax.set_zlim(zmin, zmax)
        ax.set_title(f"Heat equation FEM solution, t = {snapshots_times[k]:.3f}")

        return (surf,)

    anim = FuncAnimation(fig, update, frames=len(u_snapshots), interval=200, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)



def solve_heat_equation_backward_euler(
    nx: int = 20,
    ny: int = 20,
    lx: float = 1.0,
    ly: float = 1.0,
    T: float = 1.0,
    dt: float = 0.1,
    n_snapshots: int = 20,
):
    """
    Solve
        u_t - Delta u = f   in Omega x (0,T]
        u = g               on boundary
        u(x,0) = v(x)       in Omega

    with P1 FEM in space and Backward Euler in time on interior nodes:
        M_ii U'_i + K_ii U_i = F_i
    rewritten as
        U'_i = A U_i + b(t, U_i),   A = -M_ii^{-1} K_ii.
    """
    mesh = rectangle_mesh(nx=nx, ny=ny, lx=lx, ly=ly)
    M, K = assemble_matrices(mesh)

    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    all_nodes = np.arange(mesh.nodes.shape[0])
    bd_nodes = boundary_nodes(mesh)
    interior = np.setdiff1d(all_nodes, bd_nodes)

    Mii = M[interior][:, interior].toarray()
    Kii = K[interior][:, interior].toarray()

    A = -np.linalg.solve(Mii, Kii)

    x_int = x[interior]
    y_int = y[interior]

    def b(t: float, u: np.ndarray) -> np.ndarray:
        fvals = source_term(x_int, y_int, t)
        F = Mii @ fvals
        return np.linalg.solve(Mii, F)

    u0_full = initial_condition(x, y)
    u0 = u0_full[interior]

    res = backward_euler_solve(
        u0=u0,
        t0=0.0,
        T=T,
        h=dt,
        A=A,
        b=b,
    )

    u_full = np.zeros((res.u.shape[0], mesh.nodes.shape[0]), dtype=float)
    u_full[:, interior] = res.u

    g_bd = dirichlet_value(x[bd_nodes], y[bd_nodes], 0.0)
    u_full[:, bd_nodes] = g_bd

    max_values = np.max(np.abs(u_full), axis=1)

    return mesh, u_full, res.t, max_values

def solve_heat_etd1(
    nx: int = 20,
    ny: int = 20,
    lx: float = 1.0,
    ly: float = 1.0,
    T: float = 1.0,
    dt: float = 0.1,
):
    """
    Solve
        u_t - Delta u = f   in Omega x (0,T]
        u = g               on boundary
        u(x,0) = v(x)       in Omega
    """
    mesh = rectangle_mesh(nx=nx, ny=ny, lx=lx, ly=ly)
    M, K = assemble_matrices(mesh)

    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    all_nodes = np.arange(mesh.nodes.shape[0])
    bd_nodes = boundary_nodes(mesh)
    interior = np.setdiff1d(all_nodes, bd_nodes)

    Mii = M[interior][:, interior].toarray()
    Kii = K[interior][:, interior].toarray()

    A = -np.linalg.solve(Mii, Kii)

    x_int = x[interior]
    y_int = y[interior]

    def b(t: float, u: np.ndarray) -> np.ndarray:
        fvals = source_term(x_int, y_int, t)
        F = Mii @ fvals
        return np.linalg.solve(Mii, F)

    u0_full = initial_condition(x, y)
    u0 = u0_full[interior]

    res = etd1_solve(
        u0=u0,
        t0=0.0,
        T=T,
        h=dt,
        A=A,
        b=b,
    )

    u_full = np.zeros((res.u.shape[0], mesh.nodes.shape[0]), dtype=float)
    u_full[:, interior] = res.u

    g_bd = dirichlet_value(x[bd_nodes], y[bd_nodes], 0.0)
    u_full[:, bd_nodes] = g_bd

    max_values = np.max(np.abs(u_full), axis=1)

    return mesh, u_full, res.t, max_values



def plot_decay(times: np.ndarray, max_values: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(times, max_values, marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\max_i |U_i(t)|$")
    plt.title("Decay of the discrete solution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    output_dir = "results/fem_2d_diffusion"

    # Plot mesh
    mesh_for_plot = rectangle_mesh(nx=10, ny=10, lx=1.0, ly=1.0)
    plot_mesh(
        mesh_for_plot,
        show_node_numbers=False,
        show_element_numbers=True,
        show_element_color=False,
        save_path=os.path.join(output_dir, "mesh_plot.png"),
    )

    # Solve the heat equation
    mesh, u, times, max_values = solve_heat_equation_backward_euler(
        nx=20,
        ny=20,
        lx=1.0,
        ly=1.0,
        T=0.2,
        dt=0.001,
    )

    # snapshot_times = np.linspace(0.0, 0.1, 31)
    snapshot_times = np.logspace(0, 0, 101) / 100 -0.01
    snapshot_indices = [np.argmin(np.abs(times - t)) for t in snapshot_times]
    u_snapshots = [u[i] for i in snapshot_indices]


    animate_solution_3d(
        mesh,
        u_snapshots,
        snapshot_times,
        os.path.join(output_dir, "solution_animation_be.gif"),
    )

        # Solve the heat equation
    mesh, u, times, max_values = solve_heat_etd1(
        nx=20,
        ny=20,
        lx=1.0,
        ly=1.0,
        T=1,
        dt=0.001,
    )

    # snapshot_times = np.linspace(0.0, 0.1, 31)
    snapshot_times = np.logspace(0, 2, 101) / 100 -0.01
    snapshot_indices = [np.argmin(np.abs(times - t)) for t in snapshot_times]
    u_snapshots = [u[i] for i in snapshot_indices]


    animate_solution_3d(
        mesh,
        u_snapshots,
        snapshot_times,
        os.path.join(output_dir, "solution_animation_etd1.gif"),
    )

    # Plot time decay
    plot_decay(
        times,
        max_values,
        os.path.join(output_dir, "solution_decay.png"),
    )