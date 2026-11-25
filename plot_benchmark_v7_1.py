import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def generate_summary_figure(out_path="benchmark_v7_1_summary.png"):
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax_iter = iter(axes.flat)

    # PANEL A: Out-of-Sample 2D vs 3D Task Efficiency (n=20 fresh seeds 100..119)
    ax = next(ax_iter)
    labels = [
        "1. Naive\nBaseline",
        "2. Naive +\n1D KL Filter",
        "3. EFE:\nNo Relay",
        "4. EFE:\nNo Cull/Regen",
        "5. EFE:\nNo Epistemic",
        "6. EFE:\nBayes Inf OFF",
        "7. EFE Full\n(v7.1.0)",
        "8. Central.\nOracle"
    ]
    eff_2d = [1.91, 1.91, 2.85, 2.68, 2.32, 2.90, 2.86, 4.76]
    sd_2d = [0.11, 0.10, 0.13, 0.07, 0.14, 0.08, 0.12, 0.39]

    eff_3d = [2.55, 2.49, 2.91, 2.76, 2.57, 2.94, 2.93, 4.66]
    sd_3d = [0.18, 0.16, 0.10, 0.10, 0.16, 0.09, 0.10, 0.26]

    x = np.arange(len(labels))
    w = 0.37
    bars_2d = ax.bar(x - w / 2, eff_2d, w, yerr=sd_2d, capsize=3, label="2D Out-of-Sample (n=20, seeds 100-119)", color="#3b82f6", edgecolor="#1e3a8a", alpha=0.92)
    bars_3d = ax.bar(x + w / 2, eff_3d, w, yerr=sd_3d, capsize=3, label="3D Out-of-Sample (n=20, seeds 100-119)", color="#10b981", edgecolor="#065f46", alpha=0.92)

    for idx, rect in enumerate(bars_2d):
        if idx in (5, 6):
            rect.set_facecolor("#1d4ed8")
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#1e3a8a", fontweight="bold")

    for idx, rect in enumerate(bars_3d):
        if idx in (5, 6):
            rect.set_facecolor("#047857")
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.22, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#065f46", fontweight="bold")

    ax.annotate(
        "+49.5% (2D) / +17.5% (3D)\nvs. Honest 1D KL (n=20)",
        xy=(6, 3.12), xytext=(2.4, 4.10),
        arrowprops=dict(arrowstyle="->", color="#047857", lw=1.4),
        fontsize=8.6, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecfdf5", edgecolor="#10b981", lw=1.0)
    )

    ax.set_title("A. Out-of-Sample Noise-Gated Efficiency (Bit/J) Across Ablation Grid (n=20)", fontsize=10.8, fontweight="bold", pad=10)
    ax.set_ylabel("Noise-Gated Signal Utility (Bit / J)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 5.55)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    # PANEL B: Static Correlated Decoy Capture Rate (Trap r < 17 m, n=20)
    ax = next(ax_iter)
    trap_2d = [52.2, 49.8, 15.2, 11.6, 22.5, 15.2, 15.6, 0.3]
    trap_3d = [33.9, 32.9, 7.1, 6.4, 16.3, 8.0, 7.1, 0.1]

    ax.bar(x - w / 2, trap_2d, w, label="2D Trap Occupancy (r < 17 m, n=20)", color="#f97316", edgecolor="#9a3412", alpha=0.9)
    ax.bar(x + w / 2, trap_3d, w, label="3D Trap Occupancy (r < 17 m, n=20)", color="#ef4444", edgecolor="#991b1b", alpha=0.9)

    for idx, (t2, t3) in enumerate(zip(trap_2d, trap_3d)):
        if idx in (0, 1, 4, 5, 6, 7):
            ax.text(float(x.item(idx)) - w / 2, t2 + 1.2, f"{t2:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#9a3412", fontweight="bold")
            ax.text(float(x.item(idx)) + w / 2, t3 + 1.2, f"{t3:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#991b1b", fontweight="bold")

    ax.annotate(
        "Physical leak eta_leak=0.45\ntraps 1D KL (49.8% in 2D)",
        xy=(1, 51.0), xytext=(2.2, 56.0),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.3),
        fontsize=8.5, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#ef4444", lw=1.0)
    )

    ax.set_title("B. Correlated Decoy Capture Rate (% Agent-Steps Inside r < 17 m, n=20)", fontsize=10.8, fontweight="bold", pad=10)
    ax.set_ylabel("Trap Occupancy (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 72)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        # PANEL C: Calibration (n=5) vs Out-of-Sample (n=20) Consistency
    ax = next(ax_iter)
    c_labels = ["2D Naive+KL", "2D EFE Full", "2D Neutral Prior\n(Inf OFF)", "3D Naive+KL", "3D EFE Full", "3D Neutral Prior\n(Inf OFF)"]
    cal5 = [1.92, 2.84, 2.91, 2.42, 2.92, 2.91]
    val20 = [1.91, 2.86, 2.90, 2.49, 2.93, 2.94]

    xc5 = np.arange(len(c_labels))
    b_c5 = ax.bar(xc5 - w / 2, cal5, w, label="Calibration Split (seeds 42-46, n=5)", color="#6366f1", edgecolor="#312e81", alpha=0.9)
    b_v20 = ax.bar(xc5 + w / 2, val20, w, label="Out-of-Sample Split (seeds 100-119, n=20)", color="#14b8a6", edgecolor="#134e4a", alpha=0.9)

    for rect in b_c5:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=7.8, color="#312e81", fontweight="bold")
    for rect in b_v20:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=7.8, color="#134e4a", fontweight="bold")

    ax.set_title("C. Calibration (n=5) vs. Out-of-Sample (n=20) Split Consistency", fontsize=10.8, fontweight="bold", pad=10)
    ax.set_ylabel("Noise-Gated Signal Utility (Bit / J)", fontsize=10, fontweight="bold")
    ax.set_xticks(xc5)
    ax.set_xticklabels(c_labels, fontsize=8.3)
    ax.set_ylim(0, 3.65)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    # PANEL D: Net Component Attribution (20-Seed Out-of-Sample Delta vs EFE Full)
    ax = next(ax_iter)
    comp_labels = [
        "vs. Naive\nBaseline",
        "vs. Naive +\n1D KL Filter",
        "vs. No\nEpistemic",
        "vs. No\nCull/Regen",
        "vs. No\nCausal Relay",
        "vs. Neutral\nPrior (Inf OFF)"
    ]
    delta_2d = [0.95, 0.95, 0.54, 0.18, 0.01, -0.04]
    delta_3d = [0.38, 0.44, 0.36, 0.17, 0.02, -0.01]

    xc = np.arange(len(comp_labels))
    ax.axhline(0.0, color="#333333", linewidth=1.1)
    ax.bar(xc - w / 2, delta_2d, w, label="2D Net Gain Delta (Bit/J, n=20)", color="#3b82f6", edgecolor="#1e3a8a")
    ax.bar(xc + w / 2, delta_3d, w, label="3D Net Gain Delta (Bit/J, n=20)", color="#10b981", edgecolor="#065f46")

    for idx, (d2, d3) in enumerate(zip(delta_2d, delta_3d)):
        off2 = 0.03 if d2 >= 0 else -0.08
        off3 = 0.03 if d3 >= 0 else -0.08
        ax.text(float(xc.item(idx)) - w / 2, d2 + off2, f"{d2:+.2f}", ha="center", va="bottom", fontsize=7.8, color="#1e3a8a", fontweight="bold")
        ax.text(float(xc.item(idx)) + w / 2, d3 + off3, f"{d3:+.2f}", ha="center", va="bottom", fontsize=7.8, color="#065f46", fontweight="bold")

    ax.annotate(
        "Parity with Neutral Prior in 3D (-0.01)\n& lower trap occupancy (7.1% vs 8.0%)",
        xy=(5, -0.01), xytext=(2.2, -0.22),
        arrowprops=dict(arrowstyle="->", color="#065f46", lw=1.3),
        fontsize=8.2, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#10b981", lw=1.0)
    )

    ax.set_title("D. Component Attribution on 20 Out-of-Sample Seeds (100-119)", fontsize=10.8, fontweight="bold", pad=10)
    ax.set_ylabel("Delta Signal Utility vs. Ablated Variant (Bit / J)", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xc)
    ax.set_xticklabels(comp_labels, fontsize=8.3)
    ax.set_ylim(-0.32, 1.15)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "Swarm of Swarms (v7.1.0): Noise-Gated Zero-Leakage 2D & 3D Benchmark Under Depletion & Correlated Decoys",
        fontsize=12.0, fontweight="bold", y=0.985
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(out_path, dpi=220)
    print(f"Saved noise-gated zero-leakage publication figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate v7.1.0 summary benchmark figure")
    parser.add_argument("--out", default="benchmark_v7_1_summary.png", help="Output PNG figure path")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output PNG")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing figure '{args.out}'. Pass --overwrite or specify --out."
        )

    generate_summary_figure(args.out)


if __name__ == "__main__":
    main()
