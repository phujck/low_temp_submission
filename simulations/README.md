# Simulations

This folder contains small exact-diagonalization checks supporting the manuscript claims.

## Quick run (safe defaults)

```powershell
./run.ps1
```

## Full run (larger truncation)

```powershell
./run.ps1 -Profile full -MaxThreads 2
```

## Outputs

- `results/commuting_check.csv`
- `results/qubit_pauli_coeffs.json`

## Safety notes

The runner sets BLAS/OpenMP thread counts to `MaxThreads` before Python starts.
The default `safe` profile uses smaller bath truncations to limit CPU/RAM use.
If your system is unstable, keep `MaxThreads` at 1 and run outside VS Code.
