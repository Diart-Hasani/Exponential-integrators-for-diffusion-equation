from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.fem_edt.matrices import matrix_2x2
from src.fem_edt.manufactured import make_linear_problem
from src.fem_edt.be import backward_euler_solve


def main() -> None:
    alpha = 100.0
    t0 = 0.0
    T = 10.0
    h = 0.1

    A = matrix_2x2(alpha=alpha)
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

    plt.figure()
    plt.plot(ts, u_ex[:, 0], label="exact u1")
    plt.plot(ts, u_num[:, 0], "--", label="BE u1")
    plt.xlabel("t")
    plt.ylabel("u1(t)")
    plt.title("Component u1: Backward Euler vs exact")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/be/u1.png", dpi=300)

    plt.figure()
    plt.plot(ts, u_ex[:, 1], label="exact u2")
    plt.plot(ts, u_num[:, 1], "--", label="BE u2")
    plt.xlabel("t")
    plt.ylabel("u2(t)")
    plt.title("Component u2: Backward Euler vs exact")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/be/u2.png", dpi=300)

    plt.figure()
    plt.plot(ts, err[:, 0], label="|error u1|")
    plt.plot(ts, err[:, 1], label="|error u2|")
    plt.plot(ts, err_norm, "k:", label="||error||_2")
    plt.xlabel("t")
    plt.ylabel("error")
    plt.title("Backward Euler error vs time")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/be/error_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
