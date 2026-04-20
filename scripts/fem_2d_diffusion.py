import numpy as np

from fem.mesh_2d import rectangle_mesh, boundary_nodes
from fem.fem2d_assembly import assemble_matrices, get_eig_range_ratio


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


if __name__ == "__main__":
    nx = 10 + 1
    ny = 10
    lx = 1.0
    ly = 1.0

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

    eig_range, eig_ratio = get_eig_range_ratio(Mii, Kii)
    print("eig range of -M^{-1}K:", eig_range)
    print(
        "Theoretical range of eig: ",
        -12 / ((lx / nx) ** 2) - 12 / ((ly / ny) ** 2),
        -(np.pi**2) * ((1 / lx) ** 2 + (1 / ly) ** 2),
    )
    print("Stiffness ratio: ", eig_ratio)
    print("Theoretical stiffness ration: ", (12 * nx**2) / np.pi**2)
