import numpy as np

def matrix_2x2(alpha: float = 1.0) -> np.ndarray:
    A = np.array([[-2.0, 1.0], [1.0, -20.0]])
    return alpha * A


def check_symmetric(A: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if A is symmetric within tol."""
    return np.linalg.norm(A - A.T, ord=np.inf) <= tol


def eig_summary(A: np.ndarray) -> dict:
    """Convenience helper: eigenvalues/eigenvectors summary."""
    w, V = np.linalg.eig(A)
    return {"eigenvalues": w, "eigenvectors": V}
