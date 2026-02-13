import argparse
import csv
import os
import time
import math

import numpy as np

from suite_common import (
    SuiteProfile,
    analytic_commuting_projector_density,
    discretize_ohmic_bath,
    ensure_dir,
    exact_reduced_density_bosonic,
    expectation_value,
    fit_hmf_in_basis,
    five_level_projector_model,
    fro_error,
    hmf_from_rho,
    lambda_est_from_populations,
    lambda_from_discrete,
    lambda_ohmic_continuum,
    normalize_density,
    stochastic_path_density,
    stochastic_scalar_density,
    trace_distance,
    traceless_gauge,
    write_json,
)


PROFILES = {
    "quick": SuiteProfile(
        name="quick",
        cutoff_list=[4, 6],
        beta_list=[0.6, 0.9, 1.3, 1.8],
        coupling_list=[0.35, 0.75, 1.15],
        strong_coupling_list=[0.8, 1.3, 1.8, 2.3],
        beta_strong=2.2,
        mode_count=2,
        eta=0.18,
        omega_c=4.0,
        omega_max_factor=6.0,
        path_tau_points=24,
        path_samples=16000,
        scalar_samples=16000,
        energy_seed=20260212,
    ),
    "full": SuiteProfile(
        name="full",
        cutoff_list=[4, 6, 8],
        beta_list=[0.45, 0.65, 0.9, 1.2, 1.6, 2.1],
        coupling_list=[0.3, 0.55, 0.85, 1.15, 1.45],
        strong_coupling_list=[0.7, 1.1, 1.5, 1.9, 2.3, 2.7],
        beta_strong=2.8,
        mode_count=2,
        eta=0.18,
        omega_c=4.0,
        omega_max_factor=6.5,
        path_tau_points=32,
        path_samples=50000,
        scalar_samples=50000,
        energy_seed=20260212,
    ),
    "publish": SuiteProfile(
        name="publish",
        cutoff_list=[4, 6, 8],
        beta_list=[0.35, 0.5, 0.7, 0.95, 1.25, 1.6, 2.05, 2.6],
        coupling_list=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        strong_coupling_list=[0.65, 0.95, 1.25, 1.55, 1.85, 2.15, 2.45, 2.75],
        beta_strong=3.2,
        mode_count=3,
        eta=0.18,
        omega_c=4.0,
        omega_max_factor=7.0,
        path_tau_points=40,
        path_samples=100000,
        scalar_samples=100000,
        energy_seed=20260212,
    ),
}

REGIMES = ("cg",)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _diag_populations(rho):
    return np.real_if_close(np.diag(rho)).astype(float)


def _level_shifts(beta, energies, pops):
    p0 = max(float(pops[0]), 1e-15)
    out = []
    e0 = float(energies[0])
    for i, pi in enumerate(pops):
        pi = max(float(pi), 1e-15)
        out.append((float(energies[i]) - e0) + (1.0 / beta) * np.log(pi / p0))
    return np.asarray(out, dtype=float)


def run_lt_cg_1(profile, data_dir, rng):
    energies, hs, p, proj_idx = five_level_projector_model(seed=profile.energy_seed)
    e_ref = float(energies[0])
    e_proj = float(energies[proj_idx])
    
    N_BOOTSTRAP = 16

    rows = []
    for g in profile.coupling_list:
        omegas, cs, _ = discretize_ohmic_bath(
            num_modes=profile.mode_count,
            omega_c=profile.omega_c,
            eta=profile.eta,
            g=g,
            omega_max_factor=profile.omega_max_factor,
        )
        lam_disc = lambda_from_discrete(omegas, cs)
        lam_cont = lambda_ohmic_continuum(profile.eta, profile.omega_c, g)

        for beta in profile.beta_list:
            # Analytic is deterministic
            rho_analytic = normalize_density(analytic_commuting_projector_density(hs, p, beta, lam_disc))
            
            # Bootstrap loop for stochastic results
            p4_path_runs = []
            lam_path_runs = []
            p4_scalar_runs = []
            lam_scalar_runs = []
            
            x_var_path_runs = []
            x_var_scalar_runs = []
            
            for _ in range(N_BOOTSTRAP):
                rho_path, path_meta = stochastic_path_density(
                    energies=energies,
                    Hs=hs,
                    P=p,
                    proj_idx=proj_idx,
                    beta=beta,
                    omegas=omegas,
                    cs=cs,
                    n_tau=profile.path_tau_points,
                    n_samples=profile.path_samples,
                    rng=rng,
                )
                rho_scalar, scalar_meta = stochastic_scalar_density(
                    energies=energies,
                    Hs=hs,
                    P=p,
                    proj_idx=proj_idx,
                    beta=beta,
                    lam=lam_disc,
                    n_samples=profile.scalar_samples,
                    rng=rng,
                )
                rho_path = normalize_density(rho_path)
                rho_scalar = normalize_density(rho_scalar)
                
                pop_pa = _diag_populations(rho_path)
                pop_sc = _diag_populations(rho_scalar)
                
                p4_path_runs.append(pop_pa[proj_idx])
                p4_scalar_runs.append(pop_sc[proj_idx])
                
                lam_est_pa = lambda_est_from_populations(beta, pop_pa[0], pop_pa[proj_idx], e_ref, e_proj)
                lam_est_sc = lambda_est_from_populations(beta, pop_sc[0], pop_sc[proj_idx], e_ref, e_proj)
                
                lam_path_runs.append(lam_est_pa)
                lam_scalar_runs.append(lam_est_sc)
                
                x_var_path_runs.append(path_meta["x_var"])
                x_var_scalar_runs.append(scalar_meta["x_var"])
            
            # Statistics
            def get_stats(data):
                mu = np.mean(data)
                sem = np.std(data, ddof=1) / np.sqrt(len(data))
                return mu, sem
                
            p4_path_mu, p4_path_sem = get_stats(p4_path_runs)
            p4_scalar_mu, p4_scalar_sem = get_stats(p4_scalar_runs)
            lam_path_mu, lam_path_sem = get_stats(lam_path_runs)
            lam_scalar_mu, lam_scalar_sem = get_stats(lam_scalar_runs)
            
            x_var_path_mu, _ = get_stats(x_var_path_runs)
            x_var_scalar_mu, _ = get_stats(x_var_scalar_runs)

            pop_an = _diag_populations(rho_analytic)
            shift_an = _level_shifts(beta, energies, pop_an)
            lambda_est_analytic = lambda_est_from_populations(
                beta, pop_an[0], pop_an[proj_idx], e_ref, e_proj
            )

            for cutoff in profile.cutoff_list:
                rho_ed = normalize_density(exact_reduced_density_bosonic(hs, p, omegas, cs, cutoff, beta))
                pop_ed = _diag_populations(rho_ed)
                shift_ed = _level_shifts(beta, energies, pop_ed)
                lambda_est_ed = lambda_est_from_populations(
                    beta, pop_ed[0], pop_ed[proj_idx], e_ref, e_proj
                )

                rows.append(
                    {
                        "test_id": "LT-CG-1",
                        "beta": beta,
                        "g": g,
                        "g2": g * g,
                        "cutoff": cutoff,
                        "mode_count": profile.mode_count,
                        "energy_seed": profile.energy_seed,
                        "E0": energies[0],
                        "E1": energies[1],
                        "E2": energies[2],
                        "E3": energies[3],
                        "E4": energies[4],
                        "lambda_disc": lam_disc,
                        "lambda_cont": lam_cont,
                        # Detailed results for Fig 3
                        "p0_ed": pop_ed[0],
                        "p1_ed": pop_ed[1],
                        "p2_ed": pop_ed[2],
                        "p3_ed": pop_ed[3],
                        "p4_ed": pop_ed[proj_idx],
                        "p0_analytic": pop_an[0],
                        "p1_analytic": pop_an[1],
                        "p2_analytic": pop_an[2],
                        "p3_analytic": pop_an[3],
                        "p4_analytic": pop_an[proj_idx],
                        "shift0_ed": shift_ed[0],
                        "shift1_ed": shift_ed[1],
                        "shift2_ed": shift_ed[2],
                        "shift3_ed": shift_ed[3],
                        "shift4_ed": shift_ed[4],
                        "shift0_analytic": shift_an[0],
                        "shift1_analytic": shift_an[1],
                        "shift2_analytic": shift_an[2],
                        "shift3_analytic": shift_an[3],
                        "shift4_analytic": shift_an[4],
                        # Stochastic Summary Statistics
                        "p4_path_mean": p4_path_mu,
                        "p4_path_sem": p4_path_sem,
                        "p4_scalar_mean": p4_scalar_mu,
                        "p4_scalar_sem": p4_scalar_sem,
                        "lambda_est_ed": lambda_est_ed,
                        "lambda_est_analytic": lambda_est_analytic,
                        "lambda_est_path_mean": lam_path_mu,
                        "lambda_est_path_sem": lam_path_sem,
                        "lambda_est_scalar_mean": lam_scalar_mu,
                        "lambda_est_scalar_sem": lam_scalar_sem,
                        "x_var_path_mean": x_var_path_mu,
                        "x_var_scalar_mean": x_var_scalar_mu,
                        "c_beta_disc": 2.0 * beta * lam_disc,
                    }
                )

    write_csv(os.path.join(data_dir, "cg_lt_cg_1.csv"), rows)


def run_lt_cg_2(profile, data_dir, rng):
    energies, hs, p, proj_idx = five_level_projector_model(seed=profile.energy_seed)
    e_ref = float(energies[0])
    e_proj = float(energies[proj_idx])
    d = hs.shape[0]
    eye = np.eye(d, dtype=np.complex128)
    basis = [eye, hs, p]

    rows = []
    beta = profile.beta_strong
    cutoff = max(profile.cutoff_list)
    for g in profile.strong_coupling_list:
        omegas, cs, _ = discretize_ohmic_bath(
            num_modes=profile.mode_count,
            omega_c=profile.omega_c,
            eta=profile.eta,
            g=g,
            omega_max_factor=profile.omega_max_factor,
        )
        lam_disc = lambda_from_discrete(omegas, cs)
        lam_cont = lambda_ohmic_continuum(profile.eta, profile.omega_c, g)

        rho_ed = normalize_density(exact_reduced_density_bosonic(hs, p, omegas, cs, cutoff, beta))
        rho_analytic = normalize_density(analytic_commuting_projector_density(hs, p, beta, lam_disc))
        
        # Added stochastic calculation for Fig 2a
        rho_path, _ = stochastic_path_density(
            energies=energies,
            Hs=hs,
            P=p,
            proj_idx=proj_idx,
            beta=beta,
            omegas=omegas,
            cs=cs,
            n_tau=profile.path_tau_points,
            n_samples=profile.path_samples,
            rng=rng,
        )
        
        pop_ed = _diag_populations(rho_ed)
        pop_an = _diag_populations(rho_analytic)
        pop_path = _diag_populations(normalize_density(rho_path))

        hmf_ed = hmf_from_rho(rho_ed, beta)
        coeffs, fit_residual = fit_hmf_in_basis(hmf_ed, basis)
        hmf_ed_0 = traceless_gauge(hmf_ed)
        hmf_an_0 = traceless_gauge(hmf_from_rho(rho_analytic, beta))

        rows.append(
            {
                "test_id": "LT-CG-2",
                "beta_fixed": beta,
                "cutoff": cutoff,
                "g": g,
                "g2": g * g,
                "mode_count": profile.mode_count,
                "energy_seed": profile.energy_seed,
                "E0": energies[0],
                "E1": energies[1],
                "E2": energies[2],
                "E3": energies[3],
                "E4": energies[4],
                "lambda_disc": lam_disc,
                "lambda_cont": lam_cont,
                "lambda_est_ed": lambda_est_from_populations(
                    beta, pop_ed[0], pop_ed[proj_idx], e_ref, e_proj
                ),
                "p4_ed": pop_ed[proj_idx],
                "p4_analytic": pop_an[proj_idx],
                "p4_path": pop_path[proj_idx],
                "tr_ed": float(np.real_if_close(np.trace(rho_ed))),
                "tr_analytic": float(np.real_if_close(np.trace(rho_analytic))),
                "trace_ed_analytic": trace_distance(rho_ed, rho_analytic),
                "fro_ed_analytic": fro_error(rho_ed, rho_analytic),
                "fit_c0": float(np.real_if_close(coeffs[0])),
                "fit_a_hs": float(np.real_if_close(coeffs[1])),
                "fit_a_p": float(np.real_if_close(coeffs[2])),
                "fit_residual": fit_residual,
                "hmf_expect_ed_gauge0": expectation_value(rho_ed, hmf_ed_0),
                "hmf_expect_analytic_gauge0": expectation_value(rho_analytic, hmf_an_0),
            }
        )

    write_csv(os.path.join(data_dir, "cg_lt_cg_2.csv"), rows)


def write_manifest(out_dir, args, elapsed):
    payload = {
        "suite": "low_temp",
        "profile": args.profile,
        "regime": args.regime,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "duration_sec": elapsed,
    }
    write_json(os.path.join(out_dir, "manifest.json"), payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Run low_temp Gaussian commuting suite")
    parser.add_argument("--regime", choices=["cg", "all"], default="all")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)
    ensure_dir(data_dir)

    profile = PROFILES[args.profile]
    t0 = time.time()

    regimes = REGIMES if args.regime == "all" else (args.regime,)
    if "cg" in regimes:
        run_lt_cg_1(profile, data_dir, rng)
        run_lt_cg_2(profile, data_dir, rng)

    elapsed = time.time() - t0
    write_manifest(out_dir, args, elapsed)
    print(f"low_temp suite completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
