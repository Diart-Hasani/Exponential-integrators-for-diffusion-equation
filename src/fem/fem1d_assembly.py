import numpy as np
import scipy.sparse as sp
from fem.mesh_1d import Mesh1D


def interval_length(coords: np.ndarray) -> float:
    """
    coords: shape (2, 1), rows are [x_i]
    """
    x1 = coords[0, 0]
    x2 = coords[1, 0]

    h = abs(x2 - x1)

    if h <= 0.0:
        raise ValueError("Degenerate interval with zero length.")

    return h


def local_mass_matrix(coords: np.ndarray) -> np.ndarray:
    """
    P1 mass matrix on one 1D interval.
    """
    h = interval_length(coords)
    return (h / 6.0) * np.array(
        [[2.0, 1.0],
         [1.0, 2.0]]
    )


def local_stiffness_matrix(coords: np.ndarray) -> np.ndarray:
    """
    P1 stiffness matrix for the 1D Laplacian on one interval.
    """
    h = interval_length(coords)
    return (1.0 / h) * np.array(
        [[1.0, -1.0],
         [-1.0, 1.0]]
    )


def assemble_matrices(mesh: Mesh1D) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Assemble global mass matrix M and stiffness matrix K.
    """
    n_nodes = mesh.nodes.shape[0]

    rows = []
    cols = []
    mass_data = []
    stiff_data = []

    for elem in mesh.elements:
        coords = mesh.nodes[elem]           # shape (2, 1)
        Me = local_mass_matrix(coords)      # shape (2, 2)
        Ke = local_stiffness_matrix(coords) # shape (2, 2)

        for a in range(2):
            A = elem[a]
            for b in range(2):
                B = elem[b]

                rows.append(A)
                cols.append(B)
                mass_data.append(Me[a, b])
                stiff_data.append(Ke[a, b])

    M = sp.coo_array((mass_data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    K = sp.coo_array((stiff_data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    return M, K


def apply_dirichlet_bc_matrix_rhs(
    A: sp.csr_matrix,
    rhs: np.ndarray,
    dirichlet_nodes: np.ndarray,
    dirichlet_values: np.ndarray | float = 0.0,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Strong Dirichlet conditions:
    overwrite rows so that u_i = g_i on boundary nodes.
    """
    A = A.tolil()
    rhs = rhs.copy()

    if np.isscalar(dirichlet_values):
        g = np.full(len(dirichlet_nodes), float(dirichlet_values))
    else:
        g = np.asarray(dirichlet_values, dtype=float)

    for node, value in zip(dirichlet_nodes, g):
        A.rows[node] = [node]
        A.data[node] = [1.0]
        rhs[node] = value

    return A.tocsr(), rhs
