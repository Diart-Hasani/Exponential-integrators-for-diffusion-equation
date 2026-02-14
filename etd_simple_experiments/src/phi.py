from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import numpy as np
from scipy.linalg import expm


Array = np.ndarray


def _phi_scalar_series(z: float, k: int, terms: int = 12) -> float:
    """
    Series for phi_k(z):
      phi_k(z) = sum_{m=0}^\infty z^m / (m+k)!
    Accurate for small |z| and avoids cancellation near z=0.
    """
    denom = math.factorial(k)
    s = 1.0 / denom
    zpow = 1.0
    for m in range(1, terms):
        zpow *= z
        s += zpow / math.factorial(m + k)
    return s


def phi_scalar(z: float, k: int, small: float = 1e-8) -> float:
    """
    Compute scalar phi_k(z) stably.

    Definitions:
      phi_0(z) = exp(z)
      phi_1(z) = (exp(z) - 1) / z
      phi_{k+1}(z) = (phi_k(z) - 1/k!) / z
      with phi_k(0) = 1/k!
    """

    if abs(z) < small:
        return _phi_scalar_series(z, k)

    if k == 0:
        return math.exp(z)

    # Start from phi_0 and use recurrence up to k
    pk = math.exp(z)  # phi_0
    for j in range(0, k):
        pk = (pk - 1.0 / math.factorial(j)) / z
    return pk


def _is_symmetric(A: Array, tol: float = 1e-12) -> bool:
    return np.linalg.norm(A - A.T, ord=np.inf) <= tol


def _phi_matrix_via_eig(A: Array, h: float, ks: Iterable[int]) -> Dict[int, Array]:
    """
    Compute phi_k(hA) by diagonalization.
    Uses eigh for symmetric A, else eig.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")

    ks = sorted(set(int(k) for k in ks))

    if _is_symmetric(A):
        w, V = np.linalg.eigh(A)
        Vinv = V.T
    else:
        w, V = np.linalg.eig(A)
        Vinv = np.linalg.inv(V)

    # Build each phi_k(hA) = V diag(phi_k(h*lambda_i)) V^{-1}
    out: Dict[int, Array] = {}
    for k in ks:
        diag_entries = np.array(
            [phi_scalar(float(h * lam), k) for lam in w], dtype=float
        )
        Dk = np.diag(diag_entries)
        Mk = (V @ Dk) @ Vinv
        # Clean tiny imaginary parts from numerical noise.
        out[k] = np.real_if_close(Mk, tol=1000)
    return out


def _phi_matrix_via_block(A: Array, h: float, k_max: int) -> Dict[int, Array]:
    """
    General method (fallback): block-exponential technique.
    Requires scipy.linalg.expm.

    Constructs block matrix of size (k_max+1)*n, whose exponential contains phi_k blocks.
    """

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    m = k_max + 1
    Z = np.zeros((m * n, m * n), dtype=float)

    # Put hA on each diagonal block
    for i in range(m):
        Z[i * n : (i + 1) * n, i * n : (i + 1) * n] = h * A

    # Put identity blocks on the first superdiagonal
    iden = np.eye(n, dtype=float)
    for i in range(m - 1):
        Z[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n] = iden

    EZ = expm(Z)

    # Extract blocks: top row contains h^k * phi_k(hA)
    out: Dict[int, Array] = {}
    for k in range(m):
        block = EZ[0:n, k * n : (k + 1) * n]
        if k == 0:
            out[0] = block  # exp(hA)
        else:
            out[k] = block / (h**k)
    return out


@dataclass
class PhiCache:
    A: Array
    h: float
    method: str = "auto"  # "auto" | "eig" | "block"
    _store: Optional[Dict[int, Array]] = None

    def get(self, ks: Iterable[int]) -> Dict[int, Array]:
        ks = sorted(set(int(k) for k in ks))
        if self._store is None:
            self._store = {}

        missing = [k for k in ks if k not in self._store]
        if not missing:
            return {k: self._store[k] for k in ks}

        if self.method not in ("auto", "eig", "block"):
            raise ValueError("method must be 'auto', 'eig', or 'block'")

        use_eig = (self.method == "eig") or (self.method == "auto")
        if use_eig:
            computed = _phi_matrix_via_eig(self.A, self.h, missing)
        else:
            computed = {}

        if self.method == "block":
            kmax = max(missing)
            computed = _phi_matrix_via_block(self.A, self.h, kmax)
            computed = {k: computed[k] for k in missing}

        self._store.update(computed)
        return {k: self._store[k] for k in ks}
