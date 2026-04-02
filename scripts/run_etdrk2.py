from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.time_diff.matrices import matrix_2x2
from src.time_diff.manufactured import make_linear_problem
from src.time_diff.etdrk2 import etdrk2_solve

def main() -> None:
    alpha = 100.0
    t0 = 0.0
    T = 10.0
    h = 0.1

    A = matrix_2x2(alpha=alpha)
    problem = make_linear_problem(A)

    res = etdrk2_solve(
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

    plt.figure()
    plt.plot(ts, u_ex[:, 0], label="exact u1")
    plt.plot(ts, u_num[:, 0], "--", label="ETD1 u1")
    plt.xlabel("t")
    plt.ylabel("u1(t)")
    plt.title("Component u1: ETD1 vs exact")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/etdrk2/u1.png", dpi=300)

    plt.figure()
    plt.plot(ts, u_ex[:, 1], label="exact u2")
    plt.plot(ts, u_num[:, 1], "--", label="ETD1 u2")
    plt.xlabel("t")
    plt.ylabel("u2(t)")
    plt.title("Component u2: ETD1 vs exact")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/etdrk2/u2.png", dpi=300) 

    plt.figure()
    plt.plot(ts, err[:, 0], label="|error u1|")
    plt.plot(ts, err[:, 1], label="|error u2|")
    plt.xlabel("t")
    plt.ylabel("absolute error")
    plt.title("ETD1 absolute error vs time")
    plt.yscale("log")  # errors usually look nicer on log scale
    plt.legend()
    plt.grid(True, which="both")

    plt.savefig("results/etdrk2/error_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
