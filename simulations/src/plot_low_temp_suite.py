import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from suite_common import ensure_dir


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def _largest_cutoff_rows(arr):
    max_cutoff = np.max(arr["cutoff"])
    return arr[arr["cutoff"] == max_cutoff], int(max_cutoff)


def plot_lt_cg_1(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cg_lt_cg_1.csv"))
    arr, max_cutoff = _largest_cutoff_rows(arr)

    gs = np.unique(arr["g"])
    colors = [f"C{i % 10}" for i in range(len(gs))]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    for idx, g in enumerate(gs):
        color = colors[idx]
        sub = arr[np.isclose(arr["g"], g)]
        sub = sub[np.argsort(sub["beta"])]

        ax_a.plot(sub["beta"], sub["p4_analytic"], color=color, linewidth=2.0)
        ax_a.plot(sub["beta"], sub["p4_ed"], "o", color=color, markersize=3.7)
        ax_a.plot(sub["beta"], sub["p4_path"], "--", color=color, linewidth=1.0, alpha=0.9)
        ax_a.plot(sub["beta"], sub["p4_scalar"], ":", color=color, linewidth=1.4, alpha=0.9)

    ax_a.set_xlabel(r"$\beta$")
    ax_a.set_ylabel(r"$p_4 = \langle 4|\rho_S|4\rangle$")
    ax_a.set_title(fr"(A) Coupled-level population (cutoff={max_cutoff})")
    coupling_handles = [Line2D([0], [0], color=colors[i], lw=2.0, label=fr"$g={gs[i]:.2f}$") for i in range(len(gs))]
    method_handles = [
        Line2D([0], [0], color="k", lw=2.0, linestyle="-", label="analytic"),
        Line2D([0], [0], color="k", marker="o", lw=0.0, markersize=4, label="ED"),
        Line2D([0], [0], color="k", lw=1.0, linestyle="--", label="HS-path"),
        Line2D([0], [0], color="k", lw=1.4, linestyle=":", label="HS-scalar"),
    ]
    leg_methods = ax_a.legend(handles=method_handles, fontsize=7.5, loc="upper right", framealpha=0.9)
    ax_a.add_artist(leg_methods)
    ax_a.legend(handles=coupling_handles, fontsize=7.2, loc="center right", framealpha=0.9)

    disc_scaled = np.median(arr["lambda_disc"] / np.maximum(arr["g2"], 1e-12))
    cont_scaled = np.median(arr["lambda_cont"] / np.maximum(arr["g2"], 1e-12))
    ax_b.axhline(disc_scaled, color="k", linestyle="--", linewidth=1.3, label=r"$\lambda_{\rm disc}/g^2$")
    ax_b.axhline(cont_scaled, color="gray", linestyle="-.", linewidth=1.3, label=r"$\lambda_{\rm cont}/g^2$")

    for idx, g in enumerate(gs):
        color = colors[idx]
        sub = arr[np.isclose(arr["g"], g)]
        sub = sub[np.argsort(sub["beta"])]
        g2 = float(g * g)
        ax_b.plot(sub["beta"], sub["lambda_est_ed"] / g2, "o", color=color, markersize=3.8)
        ax_b.plot(sub["beta"], sub["lambda_est_path"] / g2, "x--", color=color, linewidth=0.95, alpha=0.9)
        ax_b.plot(sub["beta"], sub["lambda_est_scalar"] / g2, "^:", color=color, linewidth=0.95, alpha=0.9)

    ax_b.set_xlabel(r"$\beta$")
    ax_b.set_ylabel(r"$\lambda_{\mathrm{est}}/g^2$")
    ax_b.set_title(r"(B) $g^2$ scaling and $\beta$-invariance")
    b_ref_handles = [
        Line2D([0], [0], color="k", lw=1.3, linestyle="--", label=r"$\lambda_{\rm disc}/g^2$"),
        Line2D([0], [0], color="gray", lw=1.3, linestyle="-.", label=r"$\lambda_{\rm cont}/g^2$"),
    ]
    leg_refs = ax_b.legend(handles=b_ref_handles, fontsize=7.4, loc="upper right", framealpha=0.9)
    ax_b.add_artist(leg_refs)
    ax_b.legend(handles=coupling_handles, fontsize=7.2, loc="lower right", framealpha=0.9)

    # Choose the most visible renormalization point for population-level illustration.
    # In this projector model, that corresponds to the largest effective projector shift.
    row = arr[np.argmax(arr["lambda_disc"])]
    beta0 = float(row["beta"])
    g0 = float(row["g"])
    levels = np.arange(5)
    energies = np.array([row["E0"], row["E1"], row["E2"], row["E3"], row["E4"]], dtype=float)
    bare_w = np.exp(-beta0 * energies)
    bare_p = bare_w / np.sum(bare_w)
    p_ed = np.array([row["p0_ed"], row["p1_ed"], row["p2_ed"], row["p3_ed"], row["p4_ed"]], dtype=float)
    p_an = np.array([row["p0_analytic"], row["p1_analytic"], row["p2_analytic"], row["p3_analytic"], row["p4_analytic"]], dtype=float)

    width = 0.22
    ax_c.bar(levels - width, bare_p, width=width, label="bare Gibbs", color="0.75")
    ax_c.bar(levels, p_an, width=width, label="analytic HMF", color="C0")
    ax_c.bar(levels + width, p_ed, width=width, label="ED", color="C1")
    ax_c.set_xticks(levels)
    ax_c.set_xticklabels([str(i) for i in levels])
    ax_c.set_xlabel("Level index")
    ax_c.set_ylabel("Population")
    ax_c.set_title(fr"(C) Individual populations at $\beta={beta0:.2f},\,g={g0:.2f}$")
    ax_c.legend(fontsize=8, framealpha=0.9)

    shift_ed = np.array([row["shift0_ed"], row["shift1_ed"], row["shift2_ed"], row["shift3_ed"], row["shift4_ed"]], dtype=float)
    shift_an = np.array([row["shift0_analytic"], row["shift1_analytic"], row["shift2_analytic"], row["shift3_analytic"], row["shift4_analytic"]], dtype=float)
    ax_d.plot(levels, shift_an, "s--", label="analytic", color="C0")
    ax_d.plot(levels, shift_ed, "o-", label="ED", color="C1")
    ax_d.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax_d.set_xticks(levels)
    ax_d.set_xticklabels([str(i) for i in levels])
    ax_d.set_xlabel("Level index")
    ax_d.set_ylabel(r"$\Delta_i = E_i-E_0+\beta^{-1}\ln(p_i/p_0)$")
    ax_d.set_title(r"(D) Level-resolved effective shifts")
    ax_d.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("LT-CG-1: commuting Gaussian projector benchmark", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lt_cg_1.pdf"), dpi=200)
    fig.savefig(os.path.join(fig_dir, "lt_cg_1.png"), dpi=200)
    plt.close(fig)


def plot_lt_cg_2(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cg_lt_cg_2.csv"))
    arr = arr[np.argsort(arr["g2"])]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    axes[0].plot(arr["g2"], arr["p4_analytic"], "-", linewidth=2.0, label="analytic")
    axes[0].plot(arr["g2"], arr["p4_ed"], "o", markersize=5, label="ED")
    axes[0].set_xlabel(r"$g^2$")
    axes[0].set_ylabel(r"$p_4$")
    axes[0].set_title(fr"(A) Strong-coupling population at $\beta={arr['beta_fixed'][0]:.2f}$")
    axes[0].legend(fontsize=8)

    axes[1].plot(arr["g2"], arr["fit_a_p"], "o-", linewidth=1.7, label=r"fit $a_P$")
    axes[1].plot(arr["g2"], -arr["lambda_disc"], "s--", linewidth=1.5, label=r"$-\lambda_{\rm disc}$")
    axes[1].plot(arr["g2"], arr["fit_a_hs"], "k:", linewidth=1.4, label=r"fit $a_{H_S}$")
    axes[1].set_xlabel(r"$g^2$")
    axes[1].set_ylabel("Coefficient value")
    axes[1].set_title(r"(B) HMF basis coefficients in $I,H_S,P$")
    axes[1].legend(fontsize=8)

    fig.suptitle("LT-CG-2: strong-coupling interaction dominance", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lt_cg_2.pdf"), dpi=200)
    fig.savefig(os.path.join(fig_dir, "lt_cg_2.png"), dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot low_temp Gaussian commuting figures")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    plot_lt_cg_1(data_dir, fig_dir)
    plot_lt_cg_2(data_dir, fig_dir)
    print("low_temp Gaussian figures generated")


if __name__ == "__main__":
    main()
