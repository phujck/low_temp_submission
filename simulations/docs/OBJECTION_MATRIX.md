# Objection Matrix (Gaussian Commuting Suite)

| Objection | Test | Quantitative criterion |
|---|---|---|
| ED and analytic agreement is accidental at one point | LT-CG-1 | `trace_ed_analytic` remains small across full `(beta, g)` sweep and converges with cutoff |
| Stochastic implementation may not match the formal prediction | LT-CG-1 | `trace_ed_path`, `trace_ed_scalar`, and `trace_path_scalar` all remain within tolerance |
| Renormalization is not purely `g^2` | LT-CG-1, LT-CG-2 | `lambda_est/g^2` is beta-flat; linear fit of `lambda_est` vs `g^2` has positive slope and good `R^2` |
| Strong-coupling behavior may need extra operators | LT-CG-2 | ED HMF basis residual is small in `{I,H_S,P}` and fitted `a_P` tracks `-lambda_disc` |
