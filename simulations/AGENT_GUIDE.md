# Low-Temperature Gaussian Suite: Agent Guide

This repository now validates only the commuting Gaussian regime.

## Scientific Intent

The suite tests the claim that for a projector coupling operator `P` with `P^2=P`, the bath effect is a pure `g^2` renormalization:

`H_MF = H_S - lambda(g) P + const`.

The comparison is always against explicit finite-bath trace-out ED.

## Implemented Models

- System:
  - Five-level diagonal `H_S = diag(E_0,...,E_4)`.
  - `E_0=0` and `(E_1,...,E_4)` are drawn once from a seeded random distribution (`energy_seed`) and then fixed for all sweeps.
  - `P = |4><4|`.
- Bath:
  - `H_B = sum_k omega_k (a_k^dag a_k + 1/2)`.
  - `H_I = P \otimes sum_k c_k x_k`, `x_k=(a_k+a_k^dag)/sqrt(2 omega_k)`.
  - No counterterm included.
- Target spectral density:
  - `J_g(omega) = 2 eta g^2 omega exp(-omega/omega_c)`.

All states are explicitly normalized after every construction step:
`rho <- rho / Tr(rho)`.

## Tests

1. `LT-CG-1`
- Sweep in `beta` and `g`.
- Compare:
  - ED reduced state.
  - Analytic commuting prediction.
  - HS full imaginary-time path sampler.
  - HS reduced scalar sampler.
- Figure: `simulations/results/figures/lt_cg_1.pdf`.
- Data: `simulations/results/data/cg_lt_cg_1.csv`.

2. `LT-CG-2`
- Strong-coupling sweep in `g^2` at fixed large `beta`.
- Compare `p4` and fit ED-extracted HMF in basis `{I, H_S, P}`.
- Figure: `simulations/results/figures/lt_cg_2.pdf`.
- Data: `simulations/results/data/cg_lt_cg_2.csv`.

## Commands

```powershell
py -3 simulations/src/run_low_temp_suite.py --regime all --profile full --seed 42
py -3 simulations/src/plot_low_temp_suite.py
py -3 simulations/src/validate_low_temp_claims.py
```

## Output Contracts

- Data:
  - `simulations/results/data/cg_lt_cg_1.csv`
  - `simulations/results/data/cg_lt_cg_2.csv`
- Figures:
  - `simulations/results/figures/lt_cg_1.pdf`
  - `simulations/results/figures/lt_cg_2.pdf`
- Claim metrics:
  - `simulations/results/claim_metrics_low_temp.json`
- Run manifest:
  - `simulations/results/manifest.json`

## Key Diagnostics

- `trace_ed_analytic`, `trace_ed_path`, `trace_ed_scalar`.
- `lambda_est = E4 - E0 + (1/beta) ln(p4/p0)`.
- `fit_a_p` against `-lambda_disc`.
- Gauge-fixed mean-force expectation:
  - `H_MF^(0) = H_MF - Tr(H_MF)/d * I`.
  - `Tr[rho_S H_MF^(0)]`.

## Common Failure Modes

- Large ED mismatch at high coupling:
  - Increase oscillator cutoff.
- Path/scalar mismatch:
  - Increase `path_tau_points` and `path_samples`.
- Drift of `lambda_est/g^2` with `beta`:
  - Check bath discretization and population extraction.
