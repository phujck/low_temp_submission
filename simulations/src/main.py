import os
import sys
import json
import argparse


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Small exact-diagonalization checks for the Hamiltonian of mean force."
    )
    parser.add_argument(
        "--profile",
        choices=("safe", "full"),
        default=os.environ.get("HMF_PROFILE", "safe"),
        help="Parameter profile for system/bath sizes.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=int(os.environ.get("HMF_MAX_THREADS", "1")),
        help="Maximum BLAS/OpenMP threads (set before NumPy/SciPy import).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for results (default: simulations/results).",
    )
    return parser.parse_args(argv)


def set_thread_env(max_threads):
    if max_threads is None:
        return
    max_threads = max(1, int(max_threads))
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(max_threads))


def preparse_max_threads(argv):
    if "--max-threads" in argv:
        idx = argv.index("--max-threads")
        if idx + 1 < len(argv):
            try:
                return int(argv[idx + 1])
            except ValueError:
                pass
    return int(os.environ.get("HMF_MAX_THREADS", "1"))


set_thread_env(preparse_max_threads(sys.argv[1:]))

import numpy as np
from scipy.linalg import expm, logm


PROFILES = {
    "safe": {
        "commuting": {
            "H_Q_diag": [0.0, 0.8, 1.6, 2.4],
            "f_diag": [0.0, 1.0, 2.0, 3.0],
            "bath_omegas": [0.7, 1.3],
            "bath_couplings": [0.06, 0.05],
            "bath_levels": 6,
            "betas": [1.0, 2.0],
        },
        "noncommuting": {
            "H_Q_scale": 0.6,
            "bath_omega": 1.1,
            "bath_coupling": 0.3,
            "bath_levels": 6,
            "beta": 1.5,
            "f_mix": 0.3,
        },
    },
    "full": {
        "commuting": {
            "H_Q_diag": [0.0, 0.8, 1.6, 2.4],
            "f_diag": [0.0, 1.0, 2.0, 3.0],
            "bath_omegas": [0.7, 1.3, 2.1],
            "bath_couplings": [0.08, 0.06, 0.05],
            "bath_levels": 8,
            "betas": [0.5, 1.0, 2.0],
        },
        "noncommuting": {
            "H_Q_scale": 0.6,
            "bath_omega": 1.1,
            "bath_coupling": 0.3,
            "bath_levels": 8,
            "beta": 1.5,
            "f_mix": 0.3,
        },
    },
}


def destroy(n):
    a = np.zeros((n, n), dtype=np.complex128)
    for i in range(1, n):
        a[i - 1, i] = np.sqrt(i)
    return a


def oscillator_ops(n, omega, mass=1.0):
    a = destroy(n)
    adag = a.conj().T
    # H = p^2/2m + 1/2 m omega^2 x^2 = omega (a^\dag a + 1/2)
    H = omega * (adag @ a + 0.5 * np.eye(n))
    x = (a + adag) / np.sqrt(2 * mass * omega)
    return H, x


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def partial_trace(rho, dims, keep):
    # keep: list of indices to keep
    dims = list(dims)
    keep = list(keep)
    traced = [i for i in range(len(dims)) if i not in keep]

    labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if 2 * len(dims) > len(labels):
        raise ValueError("Too many subsystems for einsum-based partial trace.")

    left = labels[: len(dims)]
    right = labels[len(dims) : 2 * len(dims)]
    for i in traced:
        right[i] = left[i]

    out_labels = [left[i] for i in keep] + [right[i] for i in keep]
    expr = "".join(left + right) + "->" + "".join(out_labels)
    reshaped = rho.reshape(dims + dims)
    traced_rho = np.einsum(expr, reshaped)
    dim_keep = int(np.prod([dims[i] for i in keep]))
    return traced_rho.reshape((dim_keep, dim_keep))


def commuting_demo(out_dir, cfg):
    # System: diagonal H_Q and f (commuting case)
    H_Q = np.diag(cfg["H_Q_diag"])
    f = np.diag(cfg["f_diag"])
    f2 = f @ f

    # Bath: finite set of oscillators (truncated Hilbert space)
    omegas = np.array(cfg["bath_omegas"], dtype=float)
    couplings = np.array(cfg["bath_couplings"], dtype=float)
    n_b = int(cfg["bath_levels"])

    bath_dims = []
    H_bath_terms = []
    x_ops = []
    for w in omegas:
        H_k, x_k = oscillator_ops(n_b, w)
        bath_dims.append(n_b)
        H_bath_terms.append(H_k)
        x_ops.append(x_k)

    # Build bath Hamiltonian and coupling operator in full space
    I_sys = np.eye(H_Q.shape[0])
    I_baths = [np.eye(d) for d in bath_dims]

    # H_bath
    H_bath = np.zeros((int(np.prod(bath_dims)),) * 2, dtype=np.complex128)
    for k, H_k in enumerate(H_bath_terms):
        ops = I_baths.copy()
        ops[k] = H_k
        H_bath += kron_all(ops)

    # sum_k c_k x_k
    X_bath = np.zeros_like(H_bath)
    for k, (c_k, x_k) in enumerate(zip(couplings, x_ops)):
        ops = I_baths.copy()
        ops[k] = x_k
        X_bath += c_k * kron_all(ops)

    # Total Hamiltonian
    H_tot = np.kron(H_Q, np.eye(H_bath.shape[0])) + np.kron(I_sys, H_bath) + np.kron(f, X_bath)

    betas = list(cfg["betas"])
    rows = []
    for beta in betas:
        rho_tot = expm(-beta * H_tot)
        rho_S = partial_trace(rho_tot, [H_Q.shape[0]] + bath_dims, keep=[0])
        rho_S /= np.trace(rho_S)

        # C(beta) for discrete bath (m_k=1 convention)
        Cbeta = beta * np.sum((couplings ** 2) / (omegas ** 2))
        rho_pred = expm(-beta * H_Q + 0.5 * Cbeta * f2)
        rho_pred /= np.trace(rho_pred)

        diff = rho_S - rho_pred
        rel_err = np.linalg.norm(diff, ord='fro') / np.linalg.norm(rho_S, ord='fro')
        rows.append({"beta": beta, "rel_fro_err": rel_err, "Cbeta": Cbeta})

    out_path = os.path.join(out_dir, "commuting_check.csv")
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("beta,rel_fro_err,Cbeta\n")
        for r in rows:
            f_out.write(f"{r['beta']},{r['rel_fro_err']:.3e},{r['Cbeta']:.6f}\n")

    return rows


def pauli_decompose(H):
    # H is 2x2
    I = np.eye(2)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    coeffs = {}
    for name, op in [("I", I), ("sx", sx), ("sy", sy), ("sz", sz)]:
        coeffs[name] = float(0.5 * np.trace(op.conj().T @ H).real)
    return coeffs


def noncommuting_demo(out_dir, cfg):
    # Qubit system
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    H_Q = float(cfg["H_Q_scale"]) * sz

    # Bath: single oscillator
    omega = float(cfg["bath_omega"])
    c = float(cfg["bath_coupling"])
    n_b = int(cfg["bath_levels"])
    H_bath, x_bath = oscillator_ops(n_b, omega)

    def run_case(f, label):
        H_tot = np.kron(H_Q, np.eye(n_b)) + np.kron(np.eye(2), H_bath) + np.kron(f, c * x_bath)
        beta = float(cfg["beta"])
        rho_tot = expm(-beta * H_tot)
        rho_S = partial_trace(rho_tot, [2, n_b], keep=[0])
        rho_S /= np.trace(rho_S)
        H_eff = -(1.0 / beta) * logm(rho_S)
        H_eff = 0.5 * (H_eff + H_eff.conj().T)
        coeffs = pauli_decompose(H_eff)
        coeffs["offdiag_norm"] = float(np.hypot(coeffs["sx"], coeffs["sy"]))
        coeffs.update({"label": label, "beta": beta})
        return coeffs

    coeffs_comm = run_case(sz, "commuting_f=sz")
    f_mix = float(cfg["f_mix"])
    coeffs_noncomm = run_case(sx + f_mix * sz, f"noncommuting_f=sx+{f_mix:.2f}sz")

    out_path = os.path.join(out_dir, "qubit_pauli_coeffs.json")
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump({"commuting": coeffs_comm, "noncommuting": coeffs_noncomm}, f_out, indent=2)

    return coeffs_comm, coeffs_noncomm


def main(argv):
    args = parse_args(argv)
    base_dir = os.path.dirname(__file__)
    out_dir = args.out_dir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    os.makedirs(out_dir, exist_ok=True)

    profile = PROFILES[args.profile]
    rows = commuting_demo(out_dir, profile["commuting"])
    coeffs_comm, coeffs_noncomm = noncommuting_demo(out_dir, profile["noncommuting"])

    print("Commuting demo (normalized rho):")
    for r in rows:
        print(f"  beta={r['beta']}: rel Frobenius error={r['rel_fro_err']:.3e}, Cbeta={r['Cbeta']:.6f}")

    print("\nPauli coefficients of H_eff = -(1/beta) log rho_S:")
    print("  Commuting f=sz:", coeffs_comm)
    print("  Noncommuting:", coeffs_noncomm)


if __name__ == "__main__":
    main(sys.argv[1:])
