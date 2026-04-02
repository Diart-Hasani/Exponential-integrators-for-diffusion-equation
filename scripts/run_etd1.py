from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.time_diff.matrices import matrix_2x2
from src.time_diff.manufactured import make_semilinear_problem
from src.time_diff.etd1 import etd1_solve


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

    res = etd1_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        b=problem.b,
    )

    ts = res.t  # shape (N+1,)
    u_num = res.u  # shape (N+1, 2)
    u_ex = np.vstack([problem.u_exact(t) for t in ts])  # shape (N+1, 2)
    err = np.abs(u_num - u_ex)

    max_err = np.max(err, axis=0)
    print(f"ETD1 run: alpha={alpha}, h={h}, steps={res.n_steps}")
    print(f"Max abs error componentwise: {max_err}")
    print(f"Max abs error (overall): {np.max(err)}")


    # u1 and u2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left plot: u1
    axes[0].plot(ts, u_ex[:, 0], "k", label="exact u1")
    axes[0].plot(ts, u_num[:, 0], "--", label="ETD1 u1")
    axes[0].set_xlim([0, T])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u1(t)")
    axes[0].set_title("Component u1: ETD1 vs exact")
    axes[0].legend()
    axes[0].grid(True)

    # right plot: u2
    axes[1].plot(ts, u_ex[:, 1], "k", label="exact u2")
    axes[1].plot(ts, u_num[:, 1], "--", label="ETD1 u2")
    axes[1].set_xlim([0, T])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u2(t)")
    axes[1].set_title("Component u2: ETD1 vs exact")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig("results/etd1/u1_u2_compare.png", dpi=300)
    plt.close(fig)

    plt.figure()
    plt.plot(ts, err[:, 0], label="|error u1|")
    plt.plot(ts, err[:, 1], label="|error u2|")
    plt.xlabel("t")
    plt.ylabel("absolute error")
    plt.title("ETD1 absolute error vs time")
    plt.yscale("log")  # errors usually look nicer on log scale
    plt.legend()
    plt.grid(True, which="both")

    plt.savefig("results/etd1/error_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
