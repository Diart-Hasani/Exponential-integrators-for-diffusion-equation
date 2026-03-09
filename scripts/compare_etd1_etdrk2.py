from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.fem_edt.matrices import matrix_2x2
from src.fem_edt.manufactured import make_semilinear_problem
from src.fem_edt.etd1 import etd1_solve
from src.fem_edt.etdrk2 import etdrk2_solve


def main() -> None:
    alpha = 1.0
    ratio = 1.0
    t0 = 0.0
    T = 5.0
    h = 0.1

    A = matrix_2x2(alpha=alpha, ratio=ratio)
    # Choos kind = "oscillatory", "mixed_decay", or "stiffer_exact"
    problem = make_semilinear_problem(A, kind = "mixed_decay")

    res_etd1 = etd1_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=problem.b,
    )

    res_etdrk2 = etdrk2_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=problem.b,
    )
    ts = res_etd1.t
    u_ex = np.vstack([problem.u_exact(t) for t in ts])

    u_etd1 = res_etd1.u
    u_etdrk2 = res_etdrk2.u

    err_etd1 = np.abs(u_etd1 - u_ex)
    err_etdrk2 = np.abs(u_etdrk2 - u_ex)

    errn_etd1 = np.linalg.norm(u_etd1 - u_ex, axis=1)
    errn_etdrk2 = np.linalg.norm(u_etdrk2 - u_ex, axis=1)

    print(f"Compare ETD1 vs ETDRK2: alpha={alpha}, h={h}, steps={res_etd1.n_steps}")
    print(f"ETD1 max abs error overall: {np.max(err_etd1)}")
    print(f"ETDRK2   max abs error overall: {np.max(err_etdrk2)}")

    # u1
    plt.figure()
    plt.plot(ts, u_ex[:, 0], "k", label="exact u1")
    plt.plot(ts, u_etd1[:, 0], "--", label="ETD1 u1")
    plt.plot(ts, u_etdrk2[:, 0], ":", label="ETDRK2 u1")
    plt.xlabel("t")
    plt.ylabel("u1(t)")
    plt.title("u1: exact vs ETD1 vs ETDRK2")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/compare_etd1_etdrk2/u1_compare.png", dpi=300)

    # u2
    plt.figure()
    plt.plot(ts, u_ex[:, 1], "k", label="exact u2")
    plt.plot(ts, u_etd1[:, 1], "--", label="ETD1 u2")
    plt.plot(ts, u_etdrk2[:, 1], ":", label="ETDRK2 u2")
    plt.xlabel("t")
    plt.ylabel("u2(t)")
    plt.title("u2: exact vs ETD1 vs ETDRK2")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/compare_etd1_etdrk2/u2_compare.png", dpi=300)

    # error norms
    plt.figure()
    plt.plot(ts, errn_etd1, "--", label="||error||_2 ETD1")
    plt.plot(ts, errn_etdrk2, ":", label="||error||_2 ETDRK2")
    plt.xlabel("t")
    plt.ylabel("error norm")
    plt.title("Error norm comparison: ETD1 vs ETDRK2")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/compare_etd1_etdrk2/error_norm_compare.png", dpi=300)

    # componentwise errors
    plt.figure()
    plt.plot(ts, err_etd1[:, 0], "--", label="|e1| ETD1")
    plt.plot(ts, err_etd1[:, 1], "--", label="|e2| ETD1")
    plt.plot(ts, err_etdrk2[:, 0], ":", label="|e1| ETDRK2")
    plt.plot(ts, err_etdrk2[:, 1], ":", label="|e2| ETDRK2")
    plt.xlabel("t")
    plt.ylabel("absolute error")
    plt.title("Componentwise error: ETD1 vs ETDRK2")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/compare_etd_be/error_components_compare.png", dpi=300)


if __name__ == "__main__":
    main()
