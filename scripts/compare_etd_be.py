from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.time_diff.matrices import matrix_2x2
from src.time_diff.manufactured import make_linear_problem
from src.time_diff.etd1 import etd1_solve
from src.time_diff.be import backward_euler_solve


def main() -> None:
    alpha = 1.0
    ratio = 10.0
    t0 = 0.0
    T = 10.0
    h = 0.1

    A = matrix_2x2(alpha=alpha, ratio=ratio)
    problem = make_linear_problem(A)

    res_etd1 = etd1_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=problem.b,
    )

    res_be = backward_euler_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        fp_iters=50,
    )

    ts = res_etd1.t
    u_ex = np.vstack([problem.u_exact(t) for t in ts])

    u_etd1 = res_etd1.u
    u_be = res_be.u

    err_etd1 = np.abs(u_etd1 - u_ex)
    err_be = np.abs(u_be - u_ex)

    errn_etd1 = np.linalg.norm(u_etd1 - u_ex, axis=1)
    errn_be = np.linalg.norm(u_be - u_ex, axis=1)

    print(f"Compare ETD1 vs BE: alpha={alpha}, h={h}, steps={res_etd1.n_steps}")
    print(f"ETD1 max abs error overall: {np.max(err_etd1)}")
    print(f"BE   max abs error overall: {np.max(err_be)}")

    # u1 and u2 plots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # left plot: u1
    axes[0].plot(ts, u_ex[:, 0], "k", label="exact u1")
    axes[0].plot(ts, u_etd1[:, 0], "--", label="ETD1 u1")
    axes[0].plot(ts, u_be[:, 0], ":", label="BE u1")
    axes[0].set_xlim([0, 3])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u1(t)")
    axes[0].set_title("u1: exact vs ETD1 vs BE")
    axes[0].legend()
    axes[0].grid(True)

    # right plot: u2
    axes[1].plot(ts, u_ex[:, 1], "k", label="exact u2")
    axes[1].plot(ts, u_etd1[:, 1], "--", label="ETD1 u2")
    axes[1].plot(ts, u_be[:, 1], ":", label="BE u2")
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u2(t)")
    axes[1].set_title("u2: exact vs ETD1 vs BE")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig("results/compare_etd_be/u1_u2_compare.png", dpi=300)
    plt.close(fig)

    # error norms
    plt.figure()
    plt.plot(ts, errn_etd1, "--", label="||error||_2 ETD1")
    plt.plot(ts, errn_be, ":", label="||error||_2 BE")
    plt.xlabel("t")
    plt.ylabel("error norm")
    plt.title("Error norm comparison: ETD1 vs BE")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/compare_etd_be/error_norm_compare.png", dpi=300)

    # componentwise errors in one figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left plot: e1
    axes[0].plot(ts, err_etd1[:, 0], "--", label="|e1| ETD1")
    axes[0].plot(ts, err_be[:, 0], ":", label="|e1| BE")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("absolute error")
    axes[0].set_title("Component 1 error")
    axes[0].set_yscale("log")
    axes[0].set_ylim([10 ** (-18), 10**1])
    axes[0].legend()
    axes[0].grid(True, which="both")

    # right plot: e2
    axes[1].plot(ts, err_etd1[:, 1], "--", label="|e2| ETD1")
    axes[1].plot(ts, err_be[:, 1], ":", label="|e2| BE")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title("Component 2 error")
    axes[1].set_yscale("log")
    axes[1].set_ylim([10 ** (-18), 10**1])
    axes[1].legend()
    axes[1].grid(True, which="both")

    fig.suptitle("Componentwise error: ETD1 vs BE")
    fig.tight_layout()
    fig.savefig("results/compare_etd_be/error_components_compare.png", dpi=300)
    plt.close(fig)


def plot_error_scaling() -> None:
    alpha = 1.0
    ratio = 10.0
    t0 = 0.0
    T = 10.0
    hs = np.array([0.1, 0.01, 0.001])

    A = matrix_2x2(alpha=alpha, ratio=ratio)
    problem = make_linear_problem(A)

    errors_etd1 = np.array([])
    errors_be = np.array([])

    for h in hs:
        res_etd1 = etd1_solve(
            u0=problem.u0,
            t0=t0,
            T=T,
            h=h,
            A=A,
            b=problem.b,
        )

        res_be = backward_euler_solve(
            u0=problem.u0,
            t0=t0,
            T=T,
            h=h,
            A=A,
            fp_iters=50,
        )

        ts = res_etd1.t
        u_ex = np.vstack([problem.u_exact(t) for t in ts])

        u_etd1 = res_etd1.u
        u_be = res_be.u

        errn_etd1 = np.linalg.norm(u_etd1 - u_ex, axis=1)
        errn_be = np.linalg.norm(u_be - u_ex, axis=1)

        errors_etd1 = np.append(errors_etd1, np.max(errn_etd1))
        errors_be = np.append(errors_be, np.max(errn_be))

    # error sclaing
    plt.figure(figsize=(6, 4))
    # plt.loglog(hs, errors_etd1, "o--", label="ETD1")
    plt.loglog(hs, errors_be, "s:", label="BE")
    ref_oh = errors_be[0] * (hs / hs[0])
    plt.loglog(hs, ref_oh, "k-.", label=r"$O(h)$")
    plt.xlabel("h", fontsize=13)
    plt.ylabel(r"$\max_t \|e(t)\|_2$", fontsize=13)
    plt.title("Max error vs step size")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("results/compare_etd_be/error_scaling_be.png", dpi=300)


if __name__ == "__main__":
    main()
    plot_error_scaling()
