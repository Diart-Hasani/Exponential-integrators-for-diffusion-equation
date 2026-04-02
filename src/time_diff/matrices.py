import numpy as np

def matrix_2x2(alpha: float = 1.0, ratio: float = 10.0) -> np.ndarray:
    A = np.array([[-1.0, 0.0], [0.0, ratio * -1.0]])
    return alpha * A


def check_symmetric(A: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if A is symmetric within tol."""
    return np.linalg.norm(A - A.T, ord=np.inf) <= tol
