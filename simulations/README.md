# Simulations

This folder contains manuscript-facing numerical validation for the Gaussian commuting low-temperature paper.

## Run

```powershell
./run.ps1 -Profile full -Regime all -Seed 42
```

## Core outputs

- `results/data/cg_lt_cg_1.csv`
- `results/data/cg_lt_cg_2.csv`
- `results/figures/lt_cg_1.pdf`
- `results/figures/lt_cg_2.pdf`
- `results/claim_metrics_low_temp.json`
- `results/manifest.json`

## Scope

- Implemented in this repo version:
  - Commuting Gaussian tests only (`LT-CG-1`, `LT-CG-2`).
- Non-Gaussian and non-commuting extensions are deferred to a follow-up paper.
