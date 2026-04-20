import numpy as np

from fem.mesh_1d import interval_mesh, boundary_nodes
from fem.fem1d_assembly import assemble_matrices, get_eig_range_ratio


def initial_condition(x: np.ndarray) -> np.ndarray:
    """
    Initial value v(x,y).
    """
    ini_con = np.sin(np.pi * x)
    return ini_con


def source_term(x: np.ndarray, t: float) -> np.ndarray:
    """
    Right-hand side f(x,y,t).
    """
    return np.zeros_like(x)


def dirichlet_value(x: np.ndarray, t: float) -> np.ndarray:
    """
    Boundary data g(x,y,t) on Gamma.
    """
    return np.zeros_like(x)


def build_load_vector(mesh, t: float) -> np.ndarray:
    """
    Load vector:

    """
    x = mesh.nodes[:, 0]
    f_nodal = source_term(x, t)
    return f_nodal


if __name__ == "__main__":
    nx = 100 + 1
    lx = 1.0
    T = 0.2
    dt = 0.01

    mesh = interval_mesh(nx=nx, lx=lx)
    M, K = assemble_matrices(mesh)

    x = mesh.nodes[:, 0]

    all_nodes = np.arange(mesh.nodes.shape[0])
    bd_nodes = boundary_nodes(mesh)
    interior = np.setdiff1d(all_nodes, bd_nodes)

    Mii = M[interior][:, interior].toarray()
    Kii = K[interior][:, interior].toarray()

    eig_range, eig_ratio = get_eig_range_ratio(Mii, Kii)
    print("eig range of -M^{-1}K:", eig_range)
    print("Theoretical range of eig: ", -12 / (lx / nx) ** 2, -((np.pi / lx) ** 2))
    print("Stiffness ratio: ", eig_ratio)
    print("Theoretical stiffness ration: ", (12 * nx**2) / np.pi**2)

    x_int = x[interior]

    def b(t: float, u: np.ndarray) -> np.ndarray:
        fvals = source_term(x_int, t)
        F = Mii @ fvals
        return np.linalg.solve(Mii, F)
