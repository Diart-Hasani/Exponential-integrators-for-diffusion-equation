import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from fem.mesh_2d import Mesh2D


def triangle_area(coords: np.ndarray) -> float:
    """
    coords: shape (3, 2), rows are [x_i, y_i]
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(detJ)

    if area <= 0.0:
        raise ValueError("Degenerate triangle with zero area.")

    return area


def local_mass_matrix(coords: np.ndarray) -> np.ndarray:
    """
    P1 mass matrix on one triangle.
    """
    area = triangle_area(coords)
    return (area / 12.0) * np.array(
        [[2.0, 1.0, 1.0],
         [1.0, 2.0, 1.0],
         [1.0, 1.0, 2.0]]
    )


def local_stiffness_matrix(coords: np.ndarray) -> np.ndarray:
    """
    P1 stiffness matrix for the Laplacian on one triangle.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    area = triangle_area(coords)

    # Coefficients for gradients of hat functions
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    # grad(phi_i) = [b_i, c_i] / (2*area)
    Ke = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            Ke[i, j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)

    return Ke


def assemble_matrices(mesh: Mesh2D) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Assemble global mass matrix M and stiffness matrix K.
    """
    n_nodes = mesh.nodes.shape[0]

    rows = []
    cols = []
    mass_data = []
    stiff_data = []

    for elem in mesh.elements:
        coords = mesh.nodes[elem]           # shape (3, 2)
        Me = local_mass_matrix(coords)      # shape (3, 3)
        Ke = local_stiffness_matrix(coords) # shape (3, 3)

        for a in range(3):
            A = elem[a]   # global row index
            for b in range(3):
                B = elem[b]  # global col index

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

def get_eig_range_ratio(M: np.ndarray, K:np.ndarray) -> np.ndarray :
    lam_min = eigsh(K, M=M, k=1, which="SM", return_eigenvectors=False)[0]
    lam_max = eigsh(K, M=M, k=1, which="LM", return_eigenvectors=False)[0]

    eig_range =  np.array([-lam_max, -lam_min], dtype="float32")
    eig_ratio = lam_max/lam_min

    return eig_range, eig_ratio