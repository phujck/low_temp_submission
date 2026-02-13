import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from suite_common import ensure_dir

# Enable LaTeX rendering and set font size
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,            # Base text size
    "axes.titlesize": 10,      # Panel titles (if any)
    "axes.labelsize": 10,      # Axis labels
    "xtick.labelsize": 8,      # Tick labels
    "ytick.labelsize": 8,      # Tick labels
    "legend.fontsize": 8,      # Legend
})

def read_csv(path):
    # Depending on how csv was written, some fields might be strings if not careful?
    # np.genfromtxt handles standard numbers fine.
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def _largest_cutoff_rows(arr):
    max_cutoff = np.max(arr["cutoff"])
    return arr[arr["cutoff"] == max_cutoff], int(max_cutoff)


def add_panel_label(ax, label):
    # Position label slightly inside top-left
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')


def plot_fig1_temp(data_dir, fig_dir):
    arr1 = read_csv(os.path.join(data_dir, "cg_lt_cg_1.csv"))
    arr1, max_cutoff = _largest_cutoff_rows(arr1)
    gs = np.unique(arr1["g"])
    colors = [f"C{i % 10}" for i in range(len(gs))]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.4, 5.0))
    ax_a, ax_b = axes[0], axes[1]

    # --- Panel A: p4 vs beta ---
    for idx, g in enumerate(gs):
        color = colors[idx]
        sub = arr1[np.isclose(arr1["g"], g)]
        sub = sub[np.argsort(sub["beta"])]

        ax_a.plot(sub["beta"], sub["p4_analytic"], color=color, linewidth=1.5)
        ax_a.plot(sub["beta"], sub["p4_ed"], "o", color=color, markersize=3.0)
        # Use scalar mean (exact in commuting sector, no Trotter error)
        ax_a.plot(sub["beta"], sub["p4_scalar_mean"], "--", color=color, linewidth=0.8, alpha=0.9)

    ax_a.set_ylabel(r"$p_4$")
    add_panel_label(ax_a, "(a)")
    
    # Legend
    method_handles = [
        Line2D([0], [0], color="k", lw=1.5, linestyle="-", label="analytic"),
        Line2D([0], [0], color="k", marker="o", lw=0.0, markersize=3.0, label="ED"),
        Line2D([0], [0], color="k", lw=0.8, linestyle="--", label="stochastic"),
    ]
    ax_a.legend(handles=method_handles, fontsize=8, loc="upper right", framealpha=0.8)

    # --- Panel B: lambda vs beta (with Shaded CI) ---
    disc_scaled = np.median(arr1["lambda_disc"] / np.maximum(arr1["g2"], 1e-12))
    ax_b.axhline(disc_scaled, color="k", linestyle="--", linewidth=1.0, label=r"$\lambda_{\rm disc}/g^2$")

    for idx, g in enumerate(gs):
        color = colors[idx]
        sub = arr1[np.isclose(arr1["g"], g)]
        sub = sub[np.argsort(sub["beta"])]
        g2 = float(g * g)
        
        # ED
        ax_b.plot(sub["beta"], sub["lambda_est_ed"] / g2, "o", color=color, markersize=3.0)
        
        # Stochastic Scalar with Shaded CI
        if "lambda_est_scalar_mean" in sub.dtype.names:
            mu = sub["lambda_est_scalar_mean"] / g2
            sem = sub["lambda_est_scalar_sem"] / g2
            ax_b.plot(sub["beta"], mu, "--", color=color, linewidth=0.8)
            ax_b.fill_between(
                sub["beta"],
                mu - 1.96 * sem,
                mu + 1.96 * sem,
                color=color,
                alpha=0.2,
            )

    ax_b.set_xlabel(r"$\beta$")
    ax_b.set_ylabel(r"$\lambda_{\mathrm{est}}/g^2$")
    add_panel_label(ax_b, "(b)")
    
    # Combined legend for g values
    g_handles = [Line2D([0], [0], color=colors[i], lw=1.5, label=fr"$g={gs[i]:.2f}$") for i in range(len(gs))]
    ax_b.legend(handles=g_handles, fontsize=8, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lt_cg_fig1_temp.pdf"))
    plt.close(fig)


def plot_fig2_strong(data_dir, fig_dir):
    arr2 = read_csv(os.path.join(data_dir, "cg_lt_cg_2.csv"))
    arr2 = arr2[np.argsort(arr2["g2"])]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.4, 5.0))
    ax_c, ax_d = axes[0], axes[1]

    # --- Panel C: p4 vs g2 (Strong Coupling) ---
    ax_c.plot(arr2["g2"], arr2["p4_analytic"], "-", linewidth=1.5, label="analytic")
    ax_c.plot(arr2["g2"], arr2["p4_ed"], "o", markersize=4, label="ED")
    if "p4_path" in arr2.dtype.names:
        ax_c.plot(arr2["g2"], arr2["p4_path"], "x--", linewidth=1.0, label="stochastic")
    ax_c.set_ylabel(r"$p_4$")
    add_panel_label(ax_c, "(a)")
    ax_c.legend(fontsize=9)

    # --- Panel D: Coeffs vs g2 ---
    ax_d.plot(arr2["g2"], arr2["fit_a_p"], "o-", linewidth=1.5, label=r"Proj. coeff ($\approx -\lambda$)")
    ax_d.plot(arr2["g2"], -arr2["lambda_disc"], "s--", linewidth=1.5, label=r"Theory ($-\lambda$)")
    ax_d.plot(arr2["g2"], arr2["fit_a_hs"], "k:", linewidth=1.2, label=r"System coeff ($\approx 1$)")
    ax_d.set_xlabel(r"$g^2$")
    ax_d.set_ylabel("Coeff.")
    add_panel_label(ax_d, "(b)")
    ax_d.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lt_cg_fig2_strong.pdf"))
    plt.close(fig)


def plot_fig3_micro(data_dir, fig_dir):
    arr1 = read_csv(os.path.join(data_dir, "cg_lt_cg_1.csv"))
    arr1, max_cutoff = _largest_cutoff_rows(arr1)
    
    # Use the max lambda row for display
    row = arr1[np.argmax(arr1["lambda_disc"])]
    beta0 = float(row["beta"])
    levels = np.arange(5)
    
    # Data prep
    energies = np.array([row["E0"], row["E1"], row["E2"], row["E3"], row["E4"]], dtype=float)
    bare_w = np.exp(-beta0 * energies)
    bare_p = bare_w / np.sum(bare_w)
    
    # Check fields exist (since run_lt_cg_1 output changed)
    # They should be there if run_low_temp_suite was correct
    p_ed = np.array([row["p0_ed"], row["p1_ed"], row["p2_ed"], row["p3_ed"], row["p4_ed"]], dtype=float)
    p_an = np.array([row["p0_analytic"], row["p1_analytic"], row["p2_analytic"], row["p3_analytic"], row["p4_analytic"]], dtype=float)
    
    shift_ed = np.array([row["shift0_ed"], row["shift1_ed"], row["shift2_ed"], row["shift3_ed"], row["shift4_ed"]], dtype=float)
    shift_an = np.array([row["shift0_analytic"], row["shift1_analytic"], row["shift2_analytic"], row["shift3_analytic"], row["shift4_analytic"]], dtype=float)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.4, 5.0))
    ax_e, ax_f = axes[0], axes[1]

    # --- Panel E: Populations ---
    width = 0.22
    ax_e.bar(levels - width, bare_p, width=width, label="bare", color="0.75")
    ax_e.bar(levels, p_an, width=width, label="HMF", color="C0")
    ax_e.bar(levels + width, p_ed, width=width, label="ED", color="C1")
    ax_e.set_ylabel("Pop.")
    add_panel_label(ax_e, "(a)")
    ax_e.legend(fontsize=8, framealpha=0.8)

    # --- Panel F: Shifts ---
    ax_f.plot(levels, shift_an, "s--", label="analytic", color="C0")
    ax_f.plot(levels, shift_ed, "o-", label="ED", color="C1")
    ax_f.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax_f.set_xticks(levels)
    ax_f.set_xticklabels([str(i) for i in levels])
    ax_f.set_xlabel("State Index")
    ax_f.set_ylabel(r"$\Delta_i$")
    add_panel_label(ax_f, "(b)")
    ax_f.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lt_cg_fig3_micro.pdf"))
    plt.close(fig)


def plot_fig4_variance(data_dir, fig_dir):
    # Check if necessary columns exist
    try:
        arr1 = read_csv(os.path.join(data_dir, "cg_lt_cg_1.csv"))
    except:
        return
    
    if "x_var_path_mean" not in arr1.dtype.names:
        print("x_var_path_mean not found in data, skipping Figure 4")
        return

    arr1, _ = _largest_cutoff_rows(arr1)
    
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.0))
    ax_a, ax_b = axes[0], axes[1]
    
    # Panel A: Var vs Beta
    # Theory: Var = 2 * beta * lambda
    gs = np.unique(arr1["g"])
    
    # Select a subset of Gs for clarity if too many
    # gs = gs[1::2] if len(gs) > 4 else gs

    for i, g in enumerate(gs):
        sub = arr1[np.isclose(arr1["g"], g)]
        sub = sub[np.argsort(sub["beta"])]
        
        # lambda_disc varies with beta? No, strictly only if discretized diff?
        # lambda_disc depends on omegas/cs which depend on g, omega_c, eta. Not beta.
        # But we loop over beta.
        
        # Use first lambda_disc
        lam = sub["lambda_disc"]
        theory = 2.0 * sub["beta"] * lam
        
        ax_a.plot(sub["beta"], theory, ":", color=f"C{i}", alpha=0.6) # , label=fr"Theory" if i==0 else None)
        ax_a.plot(sub["beta"], sub["x_var_path_mean"], "o-", color=f"C{i}", markersize=4, label=fr"$g={g:.2f}$")
        
    ax_a.set_xlabel(r"$\beta$")
    ax_a.set_ylabel(r"$\sigma^2_{\xi}$")
    ax_a.legend(fontsize=8, framealpha=0.8)
    add_panel_label(ax_a, "(a)")
    
    # Panel B: Var vs g^2
    betas = np.unique(arr1["beta"])
    # Pick a few betas to avoid clutter
    betas_to_plot = betas[::2] 
    
    for i, beta in enumerate(betas_to_plot):
        sub = arr1[np.isclose(arr1["beta"], beta)]
        sub = sub[np.argsort(sub["g2"])]
        
        lam = sub["lambda_disc"]
        theory = 2.0 * beta * lam
        
        ax_b.plot(sub["g2"], theory, ":", color=f"C{i}", alpha=0.6)
        ax_b.plot(sub["g2"], sub["x_var_path_mean"], "s-", color=f"C{i}", markersize=4, label=fr"$\beta={beta:.2f}$")

    ax_b.set_xlabel(r"$g^2$")
    ax_b.set_ylabel(r"$\sigma^2_{\xi}$")
    ax_b.legend(fontsize=8, framealpha=0.8)
    add_panel_label(ax_b, "(b)")
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "lt_cg_fig4_variance.pdf"))
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

    plot_fig1_temp(data_dir, fig_dir)
    plot_fig2_strong(data_dir, fig_dir)
    plot_fig3_micro(data_dir, fig_dir)
    plot_fig4_variance(data_dir, fig_dir)
    print("low_temp Gaussian figures generated (split 1, 2, 3, 4)")


if __name__ == "__main__":
    main()
