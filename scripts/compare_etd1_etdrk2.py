from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from src.fem_edt.matrices import matrix_2x2
from src.fem_edt.manufactured import make_semilinear_problem
from src.fem_edt.etd1 import etd1_solve
from src.fem_edt.etdrk2 import etdrk2_solve


def compute_f_along_exact(problem, ts: np.ndarray) -> np.ndarray:
    """
    Compute f(t) = b(t, u_exact(t)) on the time grid.
    Returns array of shape (len(ts), d).
    """
    vals = []
    for t in ts:
        uex = problem.u_exact(float(t))
        vals.append(problem.b(float(t), uex))
    return np.vstack(vals)


def numerical_time_derivative(vals: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    Numerical derivative of a vector-valued function sampled on ts.
    vals has shape (N, d), ts has shape (N,).
    Returns array of shape (N, d).
    """
    N, d = vals.shape
    ders = np.zeros_like(vals)

    if N < 2:
        return ders

    # forward difference at left endpoint
    ders[0] = (vals[1] - vals[0]) / (ts[1] - ts[0])

    # central differences in the interior
    for n in range(1, N - 1):
        ders[n] = (vals[n + 1] - vals[n - 1]) / (ts[n + 1] - ts[n - 1])

    # backward difference at right endpoint
    ders[-1] = (vals[-1] - vals[-2]) / (ts[-1] - ts[-2])

    return ders


def main() -> None:
    alpha = 1.0
    ratio = 1.0
    t0 = 0.0
    T = 20.0
    h = 0.1

    A = matrix_2x2(alpha=alpha, ratio=ratio)
    # Choose nonlinearity = "sine" or "quadratic"
    # Choos kind = "oscillatory", "mixed_decay", "stiffer_exact" or "pure_trig"
    problem = make_semilinear_problem(A, beta=1, kind="oscillatory")

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

    # Diagnostic quantity f(t) = b(t, u_exact(t))
    f_ex = compute_f_along_exact(problem, ts)
    df_ex = numerical_time_derivative(f_ex, ts)

    abs_df1 = np.abs(df_ex[:, 0])
    abs_df2 = np.abs(df_ex[:, 1])

    print(f"max |f1'(t)| ≈ {np.max(abs_df1)}")
    print(f"max |f2'(t)| ≈ {np.max(abs_df2)}")
    print(f"Compare ETD1 vs ETDRK2: alpha={alpha}, h={h}, steps={res_etd1.n_steps}")
    print(f"ETD1 max abs error overall: {np.max(err_etd1)}")
    print(f"ETDRK2   max abs error overall: {np.max(err_etdrk2)}")

    # u1
    plt.figure()
    plt.plot(ts, u_ex[:, 0], "k", label="exact u1")
    plt.plot(ts, u_etd1[:, 0], "--", label="ETD1 u1")
    plt.plot(ts, u_etdrk2[:, 0], ":", label="ETDRK2 u1")
    plt.xlim([0,T])
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

    # derivative diagnostics
    plt.figure()
    plt.plot(ts, abs_df1, label="|f1'(t)|")
    #plt.plot(ts, abs_df2, label="|f2'(t)|")
    plt.xlim([0,T])
    plt.xlabel("t")
    plt.ylabel("magnitude")
    plt.title("Variation of f(t)=b(t,u_exact(t))")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/compare_etd1_etdrk2/fprime_components.png", dpi=300)

    # ETD1 component error vs derivative diagnostic
    plt.figure()
    plt.plot(ts, err_etd1[:, 0], "--", label="|e1| ETD1")
    #plt.plot(ts, err_etd1[:, 1], "--", label="|e2| ETD1")
    plt.plot(ts, abs_df1, ":", label="|f1'(t)|")
    #plt.plot(ts, abs_df2, ":", label="|f2'(t)|")
    plt.xlim([0,T])
    plt.xlabel("t")
    plt.ylabel("magnitude")
    plt.title("ETD1 component errors vs variation of f(t)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("results/compare_etd1_etdrk2/etd1_error_vs_fprime.png", dpi=300)


if __name__ == "__main__":
    main()
