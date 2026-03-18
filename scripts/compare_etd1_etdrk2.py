from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.etd.matrices import matrix_2x2
from src.etd.manufactured import make_semilinear_problem
from src.etd.etd1 import etd1_solve
from src.etd.etdrk2 import etdrk2_solve


def main() -> None:
    alpha = 1.0
    ratio = 1.0
    t0 = 0.0
    T = 10.0
    h = 0.1

    A = matrix_2x2(alpha=alpha, ratio=ratio)
    # Choose nonlinearity = "sine" or "quadratic"
    # Choos kind = "oscillatory", "mixed_decay", "stiffer_exact" or "pure_trig"
    problem = make_semilinear_problem(A, beta=-1, kind="oscillatory")

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

    # u1 and u2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left plot: u1
    axes[0].plot(ts, u_ex[:, 0], "k", label="exact u1")
    axes[0].plot(ts, u_etd1[:, 0], "--", label="ETD1 u1")
    axes[0].plot(ts, u_etdrk2[:, 0], ":", label="ETDRK2 u1")
    axes[0].set_xlim([0, T])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u1(t)")
    axes[0].set_title("u1: exact vs ETD1 vs ETDRK2")
    axes[0].legend()
    axes[0].grid(True)

    # right plot: u2
    axes[1].plot(ts, u_ex[:, 1], "k", label="exact u2")
    axes[1].plot(ts, u_etd1[:, 1], "--", label="ETD1 u2")
    axes[1].plot(ts, u_etdrk2[:, 1], ":", label="ETDRK2 u2")
    axes[1].set_xlim([0, T])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u2(t)")
    axes[1].set_title("u2: exact vs ETD1 vs ETDRK2")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig("results/compare_etd1_etdrk2/u1_u2_compare.png", dpi=300)
    plt.close(fig)

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

    # componentwise errors in one figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left plot: e1
    axes[0].plot(ts, err_etd1[:, 0], "--", label="|e1| ETD1")
    axes[0].plot(ts, err_etdrk2[:, 0], ":", label="|e1| ETDRK2")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("absolute error")
    axes[0].set_title("Component 1 error")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, which="both")

    # right plot: e2
    axes[1].plot(ts, err_etd1[:, 1], "--", label="|e2| ETD1")
    axes[1].plot(ts, err_etdrk2[:, 1], ":", label="|e2| ETDRK2")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title("Component 2 error")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, which="both")

    fig.suptitle("Componentwise error: ETD1 vs ETDRK2")
    fig.tight_layout()
    fig.savefig("results/compare_etd1_etdrk2/error_components_compare.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
