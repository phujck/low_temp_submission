# LT-CG-1: Beta Sweep, ED vs Analytic vs Stochastic

## Model

- System:
  - `H_S = diag(E_0, E_1, E_2, E_3, E_4)` with `E_0=0`.
  - The remaining energies are sampled once from a seeded random draw (`energy_seed`) and fixed for the full run.
  - `P = |4><4|`, `P^2 = P`.
- Bath:
  - `H_B = sum_k omega_k (a_k^dag a_k + 1/2)`.
  - `H_I = P \otimes sum_k c_k x_k`.
  - `x_k=(a_k+a_k^dag)/sqrt(2 omega_k)`.
- Ohmic target:
  - `J_g(omega)=2 eta g^2 omega exp(-omega/omega_c)`.
  - Discretized to finite modes used in ED and kernel construction.

## Exact Quantity

`rho_S^ED(beta)=Tr_B[e^{-beta H_tot}]/Tr[e^{-beta H_tot}]`.

ED is evaluated with increasing oscillator cutoff for convergence checks.
Every reported state is normalized to unit trace after construction.

## Prediction and Estimators

- Analytic commuting formula:
  - `rho_S^pred propto exp[-beta(H_S - lambda_disc P)]`.
  - `lambda_disc = sum_k c_k^2/(2 omega_k^2)`.
- Continuum reference:
  - `lambda_cont = (1/pi) int_0^inf J_g(omega)/omega d omega`.
- Stochastic estimators:
  - Full HS path sampling on an imaginary-time grid.
  - Reduced scalar sampling of `X = int_0^beta xi(tau) d tau`.

## Plotted Panels

1. Panel A:
- `p4(beta)=<4|rho_S|4>` for ED, analytic, HS-path, HS-scalar across multiple `g`.
- Claim: all methods coincide within numerical uncertainty.

2. Panel B:
- `lambda_est/g^2` vs `beta`, where
  - `lambda_est = E4-E0 + (1/beta) ln(p4/p0)`.
- Overlays horizontal `lambda_disc/g^2` and `lambda_cont/g^2`.
- Claim: beta-independence and `g^2` scaling collapse.

3. Panel C:
- Individual populations `p_i` (`i=0..4`) at representative `(beta,g)`.
- Curves/bars: bare Gibbs, analytic HMF, ED.
- Claim: the full population vector is reproduced by the HMF prediction.

4. Panel D:
- Level-resolved shift extracted from populations:
  - `Delta_i = E_i-E_0 + (1/beta) ln(p_i/p_0)`.
- Claim: `Delta_i ~ 0` for uncoupled levels and `Delta_4 ~ lambda` for the coupled level.

## Objections Addressed

- "Analytic formula may only match one temperature":
  - Answered by beta sweep for multiple couplings.
- "Path-level stochastic implementation may not match reduced scalar theory":
  - Answered by HS-path vs HS-scalar overlay and direct residuals.
- "Renormalization may not be pure `g^2`":
  - Answered by `lambda_est/g^2` collapse and linear `lambda` vs `g^2`.
