from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.time_diff.matrices import matrix_2x2
from src.time_diff.manufactured import make_linear_problem
from src.time_diff.be import backward_euler_solve


def main() -> None:
    alpha = 10
    ratio = 10
    t0 = 0.0
    T = 0.3
    h = 0.01

    A = matrix_2x2(alpha=alpha, ratio=ratio)
    problem = make_linear_problem(A)

    res = backward_euler_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        fp_iters=50,
    )

    ts = res.t
    u_num = res.u
    u_ex = np.vstack([problem.u_exact(t) for t in ts])
    err = np.abs(u_num - u_ex)
    err_norm = np.linalg.norm(u_num - u_ex, axis=1)

    max_err = np.max(err, axis=0)
    print(f"Backward Euler run: alpha={alpha}, h={h}, steps={res.n_steps}")
    print(f"Max abs error componentwise: {max_err}")
    print(f"Max abs error (overall): {np.max(err)}")

    # u1 and u2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left plot: u1
    axes[0].plot(ts, u_ex[:, 0], "k", label="exact u1")
    axes[0].plot(ts, u_num[:, 0], "--", label="BE u1")
    axes[0].set_xlim([0, T])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u1(t)")
    axes[0].set_title("Component u1: BE vs exact")
    axes[0].legend()
    axes[0].grid(True)

    # right plot: u2
    axes[1].plot(ts, u_ex[:, 1], "k", label="exact u2")
    axes[1].plot(ts, u_num[:, 1], "--", label="BE u2")
    axes[1].set_xlim([0, 0.15])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u2(t)")
    axes[1].set_title("Component u2: BE vs exact")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig("results/be/u1_u2_compare.png", dpi=300)
    plt.close(fig)

    plt.figure()
    plt.plot(ts, err[:, 0], label="|error u1|")
    plt.plot(ts, err[:, 1], label="|error u2|")
    plt.plot(ts, err_norm, "k:", label="||error||_2")
    plt.ylim([1e-4,1])
    plt.xlabel("t")
    plt.ylabel("error")
    plt.title("Backward Euler error vs time")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/be/error_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
