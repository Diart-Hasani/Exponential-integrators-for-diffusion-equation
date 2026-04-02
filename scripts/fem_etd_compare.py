import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

from fem.mesh_2d import rectangle_mesh, boundary_nodes
from fem.fem2d_assembly import assemble_matrices

from time_diff.be import backward_euler_solve
from time_diff.etd1 import etd1_solve
from time_diff.etdrk2 import etdrk2_solve


# ------------------------------------------------------------
# Problem data
# ------------------------------------------------------------
def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_term(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    # homogeneous heat equation for now
    return np.zeros_like(x)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_reduced_system(nx=20, ny=20, lx=1.0, ly=1.0):
    """
    Build mesh, assemble M and K, and restrict to interior nodes.
    """
    mesh = rectangle_mesh(nx=nx, ny=ny, lx=lx, ly=ly)
    M, K = assemble_matrices(mesh)

    all_nodes = np.arange(mesh.nodes.shape[0])
    bd = boundary_nodes(mesh)
    interior = np.setdiff1d(all_nodes, bd)

    Mii = M[interior][:, interior].toarray()
    Kii = K[interior][:, interior].toarray()

    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    return mesh, interior, bd, Mii, Kii, x, y


def make_fem_semilinear_data(Mii, Kii, interior, x, y):
    """
    Return A and b(t,u) for the reduced interior system
        U' = A U + b(t,U)
    with
        A = - Mii^{-1} Kii,
        b(t,U) = Mii^{-1} F(t)
    """
    # Dense matrix is acceptable for small thesis experiments
    A = solve(Mii, -Kii)

    x_int = x[interior]
    y_int = y[interior]

    def b(t: float, u: np.ndarray) -> np.ndarray:
        fvals = source_term(x_int, y_int, t)
        F = Mii @ fvals
        return solve(Mii, F)

    return A, b


def restrict_initial_condition(interior, x, y):
    u0_full = initial_condition(x, y)
    return u0_full[interior]


def expand_to_full_vector(u_int: np.ndarray, interior: np.ndarray, n_nodes: int) -> np.ndarray:
    u_full = np.zeros(n_nodes, dtype=float)
    u_full[interior] = u_int
    return u_full


def plot_solution(mesh, u_full, save_path, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        u_full,
        triangles=mesh.elements,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.2,
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    fig.colorbar(surf, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_decay(result, save_path, title):
    maxvals = np.max(np.abs(result.u), axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(result.t, maxvals, marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\max |U_I|$")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# Three solver runs
# ------------------------------------------------------------
def run_backward_euler(A, b, u0, t0, T, h):
    return backward_euler_solve(
        u0=u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=b,
    )


def run_etd1(A, b, u0, t0, T, h):
    return etd1_solve(
        u0=u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=b,
    )


def run_etdrk2(A, b, u0, t0, T, h):
    return etdrk2_solve(
        u0=u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=b,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    output_dir = "results/fem_compare"
    os.makedirs(output_dir, exist_ok=True)

    # FEM setup
    mesh, interior, bd, Mii, Kii, x, y = build_reduced_system(
        nx=20, ny=20, lx=1.0, ly=1.0
    )

    A, b = make_fem_semilinear_data(Mii, Kii, interior, x, y)
    u0 = restrict_initial_condition(interior, x, y)

    t0 = 0.0
    T = 0.1
    h = 0.005

    # -------------------------
    # Loop 1: Backward Euler
    # -------------------------
    be_result = run_backward_euler(A, b, u0, t0, T, h)
    u_be_final = expand_to_full_vector(
        be_result.u[-1], interior, mesh.nodes.shape[0]
    )

    print("Backward Euler finished")
    print("  steps:", be_result.n_steps)
    print("  final max |u|:", np.max(np.abs(u_be_final)))

    plot_solution(
        mesh,
        u_be_final,
        os.path.join(output_dir, "be_final.png"),
        "Backward Euler final solution",
    )
    plot_decay(
        be_result,
        os.path.join(output_dir, "be_decay.png"),
        "Backward Euler decay",
    )

    # -------------------------
    # Loop 2: ETD1
    # -------------------------
    etd1_result = run_etd1(A, b, u0, t0, T, h)
    u_etd1_final = expand_to_full_vector(
        etd1_result.u[-1], interior, mesh.nodes.shape[0]
    )

    print("ETD1 finished")
    print("  steps:", etd1_result.n_steps)
    print("  final max |u|:", np.max(np.abs(u_etd1_final)))

    plot_solution(
        mesh,
        u_etd1_final,
        os.path.join(output_dir, "etd1_final.png"),
        "ETD1 final solution",
    )
    plot_decay(
        etd1_result,
        os.path.join(output_dir, "etd1_decay.png"),
        "ETD1 decay",
    )

    # -------------------------
    # Loop 3: ETDRK2
    # -------------------------
    etdrk2_result = run_etdrk2(A, b, u0, t0, T, h)
    u_etdrk2_final = expand_to_full_vector(
        etdrk2_result.u[-1], interior, mesh.nodes.shape[0]
    )

    print("ETDRK2 finished")
    print("  steps:", etdrk2_result.n_steps)
    print("  final max |u|:", np.max(np.abs(u_etdrk2_final)))

    plot_solution(
        mesh,
        u_etdrk2_final,
        os.path.join(output_dir, "etdrk2_final.png"),
        "ETDRK2 final solution",
    )
    plot_decay(
        etdrk2_result,
        os.path.join(output_dir, "etdrk2_decay.png"),
        "ETDRK2 decay",
    )