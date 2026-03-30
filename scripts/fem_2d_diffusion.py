import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.sparse.linalg import spsolve

from fem.mesh_2d import rectangle_mesh, plot_mesh, boundary_nodes
from fem.fem2d_assembly import assemble_matrices, apply_dirichlet_bc_matrix_rhs


def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Initial value v(x,y).
    """
    # ini_con = np.cos(np.pi * x) * np.sin(np.pi * y)
    ini_con = 0.1*x + np.ones_like(y)
    #
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

    def update(k):
        ax.clear()

        surf = ax.plot_trisurf(
            mesh.nodes[:, 0],
            mesh.nodes[:, 1],
            u_snapshots[k],
            triangles=mesh.elements,
            cmap="viridis",
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
    anim.save(save_path, writer=PillowWriter(fps=20))
    plt.close(fig)



def solve_heat_equation_backward_euler(
    nx: int = 20,
    ny: int = 20,
    lx: float = 1.0,
    ly: float = 1.0,
    T: float = 1,
    dt: float = 0.1,
    n_snapshots: int = 20,
):
    """
    Solve
        u_t - Delta u = f   in Omega x (0,T]
        u = g               on boundary
        u(x,0) = v(x)       in Omega

    with P1 FEM in space and Backward Euler in time:
        (M + dt K) U^{n+1} = M U^n + dt F^{n+1}
    """

    # Mesh and matrices
    mesh = rectangle_mesh(nx=nx, ny=ny, lx=lx, ly=ly)
    M, K = assemble_matrices(mesh)

    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    bd_nodes = boundary_nodes(mesh)

    # Initial condition
    U = initial_condition(x, y)

    # Enforce boundary values at t=0
    g0 = dirichlet_value(x[bd_nodes], y[bd_nodes], t=0.0)
    U[bd_nodes] = g0

    # Time-stepping setup
    n_steps = int(round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)

    # For saving evolution info
    max_values = [np.max(np.abs(U))]

    # Snapshots
    snapshot_indices = np.linspace(0, n_steps, n_snapshots, dtype=int)

    u_snapshots = [U.copy()]
    snapshot_times = [0.0]

    # Backward Euler loop
    for n in range(n_steps):
        t_np1 = times[n + 1]

        # Approximate load vector.
        # Better later: assemble true FEM load vector elementwise.
        f_nodal = build_load_vector(mesh, t_np1)
        F_np1 = M @ f_nodal

        # System:
        # (M + dt K) U^{n+1} = M U^n + dt F^{n+1}
        A = M + dt * K
        rhs = M @ U + dt * F_np1

        # Dirichlet boundary values at time t_{n+1}
        g_np1 = dirichlet_value(x[bd_nodes], y[bd_nodes], t_np1)

        # Strongly impose BC
        A_bc, rhs_bc = apply_dirichlet_bc_matrix_rhs(A, rhs, bd_nodes, g_np1)

        # Solve linear system
        U = spsolve(A_bc, rhs_bc)

        if (n + 1) in snapshot_indices:
            u_snapshots.append(U.copy())
            snapshot_times.append(t_np1)

        max_values.append(np.max(np.abs(U)))

    return mesh, u_snapshots, np.array(snapshot_times), times, np.array(max_values)


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
    mesh, u_snapshots, snapshot_times, times, max_values = solve_heat_equation_backward_euler(
        nx=20,
        ny=20,
        lx=1.0,
        ly=1.0,
        T=0.1,
        dt=0.001,
        n_snapshots=100,
    )

    animate_solution_3d(
        mesh,
        u_snapshots,
        snapshot_times,
        os.path.join(output_dir, "solution_animation.gif"),
    )


    # Plot time decay
    plot_decay(
        times,
        max_values,
        os.path.join(output_dir, "solution_decay.png"),
    )