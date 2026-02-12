import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, logm


EPS = 1e-12


@dataclass
class SuiteProfile:
    name: str
    cutoff_list: list
    beta_list: list
    coupling_list: list
    strong_coupling_list: list
    beta_strong: float
    mode_count: int
    eta: float
    omega_c: float
    omega_max_factor: float
    path_tau_points: int
    path_samples: int
    scalar_samples: int
    energy_seed: int


def ensure_dir(path):
    import os

    os.makedirs(path, exist_ok=True)


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def partial_trace(rho, dims, keep):
    dims = list(dims)
    keep = list(keep)
    traced = [i for i in range(len(dims)) if i not in keep]
    labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if 2 * len(dims) > len(labels):
        raise ValueError("Too many subsystems for einsum trace.")
    left = labels[: len(dims)]
    right = labels[len(dims) : 2 * len(dims)]
    for idx in traced:
        right[idx] = left[idx]
    out_labels = [left[i] for i in keep] + [right[i] for i in keep]
    expr = "".join(left + right) + "->" + "".join(out_labels)
    traced_rho = np.einsum(expr, rho.reshape(dims + dims))
    dim_keep = int(np.prod([dims[i] for i in keep]))
    return traced_rho.reshape((dim_keep, dim_keep))


def hermitize(a):
    return 0.5 * (a + a.conj().T)


def normalize_density(rho):
    rho = hermitize(np.asarray(rho, dtype=np.complex128))
    tr = np.trace(rho)
    if abs(tr) < EPS:
        raise ValueError("Cannot normalize matrix with zero trace.")
    rho = rho / tr
    tr2 = np.trace(rho)
    if abs(tr2) < EPS:
        raise ValueError("Normalization failed due to near-zero trace.")
    return hermitize(rho / tr2)


def thermal_density(H, beta):
    H = hermitize(np.asarray(H, dtype=np.complex128))
    evals = np.linalg.eigvalsh(H)
    shift = float(np.min(np.real(evals)))
    return expm(-beta * (H - shift * np.eye(H.shape[0], dtype=np.complex128)))


def hmf_from_rho(rho, beta):
    return hermitize(-(1.0 / beta) * logm(rho))


def trace_distance(rho_a, rho_b):
    vals = np.linalg.eigvalsh(hermitize(rho_a - rho_b))
    return float(0.5 * np.sum(np.abs(vals)))


def fro_error(rho_a, rho_b):
    return float(np.linalg.norm(rho_a - rho_b, ord="fro"))


def expectation_value(rho, op):
    return float(np.real_if_close(np.trace(rho @ op)))


def traceless_gauge(op):
    d = op.shape[0]
    return hermitize(op - (np.trace(op) / d) * np.eye(d, dtype=np.complex128))


def fit_hmf_in_basis(hmf, basis_ops):
    vecs = [op.reshape(-1) for op in basis_ops]
    a = np.stack(vecs, axis=1)
    b = hmf.reshape(-1)
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    recon = sum(coeffs[i] * basis_ops[i] for i in range(len(basis_ops)))
    resid = np.linalg.norm(hmf - recon, ord="fro") / max(np.linalg.norm(hmf, ord="fro"), EPS)
    return coeffs, float(np.real_if_close(resid))


def destroy(n):
    a = np.zeros((n, n), dtype=np.complex128)
    for i in range(1, n):
        a[i - 1, i] = math.sqrt(i)
    return a


def oscillator_ops(n, omega):
    a = destroy(n)
    adag = a.conj().T
    h = omega * (adag @ a + 0.5 * np.eye(n, dtype=np.complex128))
    x = (a + adag) / math.sqrt(2.0 * omega)
    return h, x


def exact_reduced_density(Hs, Hb, Hi, beta):
    ds = Hs.shape[0]
    db = Hb.shape[0]
    htot = np.kron(Hs, np.eye(db, dtype=np.complex128)) + np.kron(np.eye(ds, dtype=np.complex128), Hb) + Hi
    rho_tot = thermal_density(htot, beta)
    rho_s = partial_trace(rho_tot, [ds, db], keep=[0])
    return normalize_density(rho_s)


def exact_reduced_density_bosonic(Hs, f, bath_omegas, bath_cs, cutoff, beta):
    h_b_terms = []
    x_terms = []
    for w in bath_omegas:
        hk, xk = oscillator_ops(cutoff, w)
        h_b_terms.append(hk)
        x_terms.append(xk)
    eye_b = [np.eye(cutoff, dtype=np.complex128) for _ in bath_omegas]
    dim_b = cutoff ** len(bath_omegas)
    h_b = np.zeros((dim_b, dim_b), dtype=np.complex128)
    x_b = np.zeros((dim_b, dim_b), dtype=np.complex128)
    for i, hk in enumerate(h_b_terms):
        mats = list(eye_b)
        mats[i] = hk
        h_b = h_b + kron_all(mats)
    for i, xk in enumerate(x_terms):
        mats = list(eye_b)
        mats[i] = bath_cs[i] * xk
        x_b = x_b + kron_all(mats)
    hi = np.kron(f, x_b)
    return exact_reduced_density(Hs, h_b, hi, beta)


def five_level_projector_model(seed=20260212):
    rng = np.random.default_rng(int(seed))
    raw = np.sort(rng.uniform(0.2, 2.6, size=4))
    min_gap = 0.1
    for i in range(1, raw.size):
        if raw[i] - raw[i - 1] < min_gap:
            raw[i] = raw[i - 1] + min_gap
    energies = np.concatenate(([0.0], raw)).astype(float)
    hs = np.diag(energies).astype(np.complex128)
    p = np.zeros((5, 5), dtype=np.complex128)
    p[-1, -1] = 1.0
    return energies, hs, p, hs.shape[0] - 1


def ohmic_spectral_density(omega, eta, omega_c, g):
    omega = np.asarray(omega, dtype=float)
    return 2.0 * eta * (g * g) * omega * np.exp(-omega / omega_c)


def lambda_ohmic_continuum(eta, omega_c, g):
    return float((2.0 * eta * omega_c / np.pi) * (g * g))


def discretize_ohmic_bath(num_modes, omega_c, eta, g, omega_max_factor=6.0):
    omega_max = float(omega_max_factor * omega_c)
    edges = np.linspace(0.0, omega_max, num_modes + 1)
    widths = np.diff(edges)
    omegas = 0.5 * (edges[:-1] + edges[1:])
    jvals = ohmic_spectral_density(omegas, eta, omega_c, g)
    c2 = (2.0 / np.pi) * jvals * omegas * widths
    cs = np.sqrt(np.maximum(c2, 0.0))
    return omegas, cs, widths


def lambda_from_discrete(omegas, cs):
    return float(np.sum((cs * cs) / (2.0 * omegas * omegas)))


def analytic_commuting_projector_density(Hs, P, beta, lam):
    rho = thermal_density(Hs - lam * P, beta)
    return normalize_density(rho)


def _logmeanexp(samples):
    smax = float(np.max(samples))
    return smax + math.log(float(np.mean(np.exp(samples - smax))))


def populations_from_shift(energies, proj_idx, beta, shift):
    weights = np.exp(-beta * energies)
    weights = weights.astype(float)
    weights[proj_idx] *= math.exp(beta * shift)
    return weights / np.sum(weights)


def density_from_populations(pop):
    return normalize_density(np.diag(np.asarray(pop, dtype=np.complex128)))


def projector_populations_from_X_samples(energies, proj_idx, beta, x_samples):
    base = np.exp(-beta * energies).astype(float)
    pop_unnorm = base.copy()
    pop_unnorm[proj_idx] = base[proj_idx] * math.exp(_logmeanexp(x_samples))
    return pop_unnorm / np.sum(pop_unnorm)


def stochastic_scalar_density(energies, Hs, P, proj_idx, beta, lam, n_samples, rng):
    del Hs, P
    c_beta = 2.0 * beta * lam
    x_samples = math.sqrt(max(c_beta, EPS)) * rng.standard_normal(int(n_samples))
    pop = projector_populations_from_X_samples(energies, proj_idx, beta, x_samples)
    return normalize_density(density_from_populations(pop)), {
        "x_mean": float(np.mean(x_samples)),
        "x_var": float(np.var(x_samples)),
        "c_beta": float(c_beta),
    }


def kernel_discrete(tau, beta, omegas, cs):
    tau = np.asarray(tau, dtype=float)
    coeff = (cs * cs) / (2.0 * omegas)
    denom = np.sinh(0.5 * beta * omegas)
    denom = np.maximum(denom, EPS)
    out = np.zeros_like(tau, dtype=float)
    for k in range(len(omegas)):
        out += coeff[k] * np.cosh(omegas[k] * (0.5 * beta - np.abs(tau))) / denom[k]
    return out


def build_tau_covariance(beta, n_tau, omegas, cs):
    dt = beta / float(n_tau)
    taus = (np.arange(n_tau, dtype=float) + 0.5) * dt
    cov = np.zeros((n_tau, n_tau), dtype=float)
    for i in range(n_tau):
        for j in range(i, n_tau):
            d = abs(taus[i] - taus[j])
            d = min(d, beta - d)
            kij = float(kernel_discrete(np.array([d]), beta, omegas, cs)[0])
            cov[i, j] = kij
            cov[j, i] = kij
    return taus, dt, cov


def sample_gaussian_paths(cov, n_samples, rng):
    evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
    evals = np.maximum(evals, 0.0)
    root = evecs @ np.diag(np.sqrt(evals))
    z = rng.standard_normal((int(n_samples), cov.shape[0]))
    return z @ root.T


def stochastic_path_density(energies, Hs, P, proj_idx, beta, omegas, cs, n_tau, n_samples, rng):
    del Hs, P
    _, dt, cov = build_tau_covariance(beta, n_tau, omegas, cs)
    xi_paths = sample_gaussian_paths(cov, n_samples, rng)
    x_samples = dt * np.sum(xi_paths, axis=1)
    pop = projector_populations_from_X_samples(energies, proj_idx, beta, x_samples)
    c_beta_grid = float(dt * dt * np.sum(cov))
    return normalize_density(density_from_populations(pop)), {
        "x_mean": float(np.mean(x_samples)),
        "x_var": float(np.var(x_samples)),
        "c_beta_grid": c_beta_grid,
    }


def lambda_est_from_populations(beta, p_ref, p_proj, e_ref, e_proj):
    p_ref = max(float(p_ref), EPS)
    p_proj = max(float(p_proj), EPS)
    return float((e_proj - e_ref) + (1.0 / beta) * math.log(p_proj / p_ref))


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    raise TypeError(f"Unsupported type: {type(obj)}")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_jsonable)

