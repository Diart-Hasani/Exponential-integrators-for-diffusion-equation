from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from etd_simple_experiments.src.matrices import matrix_2x2
from etd_simple_experiments.src.manufactured import make_stiff_linear_problem


def backward_euler_solve(
    u0: np.ndarray,
    t0: float,
    T: float,
    h: float,
    A: np.ndarray,
    fp_iters: int = 20,
    tol: float = 1e-12,
):
    """
    Solve u' = A u with backward Euler:
        u_{n+1} = u_n + h A u_{n+1}
    using fixed-point iteration.
    """
    u0 = np.asarray(u0, dtype=float)
    A = np.asarray(A, dtype=float)
    d = u0.size

    times = [float(t0)]
    us = [u0.copy()]

    t = float(t0)
    u = u0.copy()

    id = np.eye(d)

    while t < T - 1e-15:
        h_step = min(h, T - t)
        t_next = t + h_step

        # Rearranged iteration:
        # (I - hA) u^{k+1} = u_n + h b(t_{n+1}, u^k)
        M = id - h_step * A

        # Initial guess: previous value (or explicit Euler predictor)
        u_guess = u.copy()

        for _ in range(fp_iters):
            rhs = u + h_step
            u_new = np.linalg.solve(M, rhs)

            if np.linalg.norm(u_new - u_guess, ord=np.inf) < tol:
                u_guess = u_new
                break
            u_guess = u_new

        u = u_guess
        t = t_next

        times.append(float(t))
        us.append(u.copy())

    class Result:
        pass

    res = Result()
    res.t = np.array(times, dtype=float)
    res.u = np.vstack(us)
    res.h = h
    res.n_steps = len(times) - 1
    return res


def main() -> None:
    # Make A stiff so ETD vs BE behavior is visible
    alpha = 100.0
    t0 = 0.0
    T = 10.0

    # Try a few h values later (e.g. 0.1, 0.5, 1.0) to compare robustness/accuracy
    h = 0.1

    A = matrix_2x2(alpha=alpha)
    problem = make_stiff_linear_problem(A)

    res = backward_euler_solve(
        u0=problem.u0,
        t0=t0,
        T=T,
        h=h,
        A=A,
        fp_iters=50,
        tol=1e-12,
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
    print(f"Max l2 error over time: {np.max(err_norm)}")

    plt.figure()
    plt.plot(ts, u_ex[:, 0], label="exact u1")
    plt.plot(ts, u_num[:, 0], "--", label="BE u1")
    plt.xlabel("t")
    plt.ylabel("u1(t)")
    plt.title("Component u1: Backward Euler vs exact")
    plt.legend()
    plt.grid(True)
    plt.savefig("etd_simple_experiments/results/back_eul/u1.png", dpi=300)

    plt.figure()
    plt.plot(ts, u_ex[:, 1], label="exact u2")
    plt.plot(ts, u_num[:, 1], "--", label="BE u2")
    plt.xlabel("t")
    plt.ylabel("u2(t)")
    plt.title("Component u2: Backward Euler vs exact")
    plt.legend()
    plt.grid(True)
    plt.savefig("etd_simple_experiments/results/back_eul/u2.png", dpi=300)

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
    plt.savefig("etd_simple_experiments/results/back_eul/error_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
