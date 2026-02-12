# LT-CG-2: Strong-Coupling Projector Dominance

## Model

Same Hamiltonian family as `LT-CG-1`:

- `H_S = diag(E_0, E_1, E_2, E_3, E_4)` with seeded random nondegenerate diagonal energies.
- `P = |4><4|`.
- `H_I = P \otimes sum_k c_k x_k`.

The sweep is at fixed large `beta`, varying `g` into strong coupling.

## Exact Quantity

`rho_S^ED(beta)=Tr_B[e^{-beta H_tot}]/Tr[e^{-beta H_tot}]` at maximal cutoff in profile.
States are normalized to unit trace before any observable extraction.

## Prediction and Extraction

- Analytic:
  - `rho_S^pred propto exp[-beta(H_S - lambda_disc P)]`.
- ED-extracted mean-force operator:
  - `H_MF^ED = -(1/beta) log(rho_S^ED)`.
- Basis fit:
  - `H_MF^ED ~ c0 I + a_HS H_S + a_P P`.

## Plotted Panels

1. Panel A:
- `p4` vs `g^2` (ED and analytic).
- Claim: projector-level population trend matches exact commuting prediction across strong coupling.

2. Panel B:
- `a_P` and `-lambda_disc` vs `g^2`, plus `a_HS`.
- Claim: interaction coefficient tracks `-lambda_disc` and shows increasing projector contribution.

## Objections Addressed

- "Agreement may only hold perturbatively":
  - Strong-coupling sweep checks beyond weak coupling.
- "HMF fit might require hidden operators":
  - Basis residual quantifies closure quality in `{I, H_S, P}`.
