import argparse
import os

import numpy as np

from suite_common import ensure_dir, write_json


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
    return slope, intercept, r2


def validate(data_dir):
    cg1 = read_csv(os.path.join(data_dir, "cg_lt_cg_1.csv"))
    cg2 = read_csv(os.path.join(data_dir, "cg_lt_cg_2.csv"))

    max_cutoff = np.max(cg1["cutoff"])
    cg1_max = cg1[cg1["cutoff"] == max_cutoff]

    scaled_lambda_ed = cg1_max["lambda_est_ed"] / np.maximum(cg1_max["g2"], 1e-12)
    collapse_by_g = []
    for g in np.unique(cg1_max["g"]):
        sub = cg1_max[np.isclose(cg1_max["g"], g)]
        vals = sub["lambda_est_ed"] / max(g * g, 1e-12)
        collapse_by_g.append(float(np.max(np.abs(vals - np.mean(vals)))))
    collapse_max_dev = float(max(collapse_by_g) if collapse_by_g else 0.0)

    slope_lambda, intercept_lambda, r2_lambda = linear_fit(cg2["g2"], cg2["lambda_est_ed"])
    slope_ap, intercept_ap, r2_ap = linear_fit(cg2["g2"], cg2["fit_a_p"])

    proj_coeff_err = np.max(np.abs(cg2["fit_a_p"] + cg2["lambda_disc"]))
    hmf_resid = float(np.max(cg2["fit_residual"]))

    max_trace_ed_an = float(np.max(cg1_max["trace_ed_analytic"]))
    max_trace_ed_path = float(np.max(cg1_max["trace_ed_path"]))
    max_trace_ed_scalar = float(np.max(cg1_max["trace_ed_scalar"]))
    max_norm_dev = 0.0
    for key in ("tr_ed", "tr_analytic", "tr_path", "tr_scalar"):
        if key in cg1_max.dtype.names:
            max_norm_dev = max(max_norm_dev, float(np.max(np.abs(cg1_max[key] - 1.0))))
    for key in ("tr_ed", "tr_analytic"):
        if key in cg2.dtype.names:
            max_norm_dev = max(max_norm_dev, float(np.max(np.abs(cg2[key] - 1.0))))

    uncoupled_shift_max = 0.0
    coupled_shift_err = 0.0
    if all(k in cg1_max.dtype.names for k in ("shift1_ed", "shift2_ed", "shift3_ed", "shift4_ed", "lambda_est_ed")):
        uncoupled_shift_max = float(
            np.max(
                np.abs(
                    np.r_[cg1_max["shift1_ed"], cg1_max["shift2_ed"], cg1_max["shift3_ed"]]
                )
            )
        )
        coupled_shift_err = float(np.max(np.abs(cg1_max["shift4_ed"] - cg1_max["lambda_est_ed"])))

    metrics = {
        "trace_distance": max_trace_ed_an,
        "fro_error": float(np.max(cg1_max["fro_ed_analytic"])),
        "hmf_basis_residual": hmf_resid,
        "collapse_max_deviation": collapse_max_dev,
        "slope_fit": {
            "lambda_est_vs_g2": {
                "slope": slope_lambda,
                "intercept": intercept_lambda,
                "r2": r2_lambda,
            },
            "aP_vs_g2": {
                "slope": slope_ap,
                "intercept": intercept_ap,
                "r2": r2_ap,
            },
        },
        "improvement_ratio": float(np.median((cg1_max["trace_ed_path"] + 1e-12) / (cg1_max["trace_ed_scalar"] + 1e-12))),
        "trace_distance_path": max_trace_ed_path,
        "trace_distance_scalar": max_trace_ed_scalar,
        "projector_coeff_max_abs_err": float(proj_coeff_err),
        "normalization_max_deviation": float(max_norm_dev),
        "uncoupled_shift_max": float(uncoupled_shift_max),
        "coupled_shift_err": float(coupled_shift_err),
    }

    checks = []
    checks.append(max_trace_ed_an < 5.0e-2)
    checks.append(max_trace_ed_path < 1.0e-1)
    checks.append(max_trace_ed_scalar < 1.0e-1)
    checks.append(collapse_max_dev < 5.0e-2)
    checks.append(hmf_resid < 8.0e-2)
    checks.append(abs(proj_coeff_err) < 2.0e-1)
    checks.append(max_norm_dev < 1.0e-10)
    checks.append(uncoupled_shift_max < 1.5e-1)
    checks.append(coupled_shift_err < 1.5e-1)
    checks.append(slope_lambda > 0.0)
    checks.append(slope_ap < 0.0)

    metrics["pass_fail"] = bool(all(checks))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Validate low_temp Gaussian commuting claims")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)

    metrics = validate(data_dir)
    write_json(os.path.join(out_dir, "claim_metrics_low_temp.json"), metrics)

    status = "PASS" if metrics["pass_fail"] else "FAIL"
    print(f"low_temp validation: {status}")
    print(metrics)


if __name__ == "__main__":
    main()
