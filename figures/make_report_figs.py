"""Report figures: label distribution and hedge prevalence by pathology.

Numbers are pre-computed from the cohort parquet files and baked in here
because these summary statistics do not change after cohort construction.
The figures use the same paper-style palette and axis treatment as
make_paper_charts.py (box border, inward ticks, bold value labels).

  python figures/make_report_figs.py --out figs/
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# retro olive/rust/mustard palette shared with the poster
GREEN = "#4A5D3A"    # positive
RUST = "#B5651D"     # negative
MUSTARD = "#CC9A3A"  # uncertain
INK = "#1A1A1A"
GRID = "#D8D3C4"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["text.color"] = INK
plt.rcParams["axes.edgecolor"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK
plt.rcParams["axes.linewidth"] = 1.4

# x positions and bar width shared between both figures
GROUPS = ["Pleural\neffusion", "Pulmonary\nedema", "Consolidation", "Pneumonia"]
X = [0, 1.4, 2.8, 4.2]
W = 0.34


def paper_axes(ax):
    """Classic journal look: box on all 4 sides, inward ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.4)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, length=5, width=1.2, labelsize=13)
    ax.set_axisbelow(True)


def _legend_handles():
    return [
        Patch(facecolor=RUST, edgecolor=INK, label="negative"),
        Patch(facecolor=MUSTARD, edgecolor=INK, label="uncertain"),
        Patch(facecolor=GREEN, edgecolor=INK, label="positive"),
    ]


def save(fig, stem, out_dir):
    """Write PNG + PDF to out_dir, then close the figure."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}",
                    transparent=True, bbox_inches="tight")
    plt.close(fig)


def make_label_dist(out_dir):
    """Grouped bar chart: label-class share of each pathology cohort.
    Numbers from build_cohort() output; n totals shown on x-axis labels."""
    neg = [24.6, 22.7, 40.2, 8.7]
    unc = [5.4, 14.9, 39.7, 74.8]
    pos = [70.0, 62.4, 20.1, 16.5]
    ns  = [90016, 58561, 46317, 17888]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for xi, a, b, c in zip(X, neg, unc, pos):
        for bx, val, col in [(xi - W, a, RUST), (xi, b, MUSTARD), (xi + W, c, GREEN)]:
            ax.bar(bx, val, width=W, color=col, edgecolor=INK, linewidth=1.1, zorder=3)
            ax.text(bx, val + 1.3, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK, zorder=4)

    ax.set_xticks(X)
    ax.set_xticklabels([f"{g}\n$n$={n:,}" for g, n in zip(GROUPS, ns)], fontsize=12.5)
    ax.set_ylim(0, 86)
    ax.set_ylabel("share of cohort (%)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    paper_axes(ax)
    ax.legend(handles=_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, fontsize=13, frameon=False, columnspacing=1.4, handletextpad=0.5)

    plt.tight_layout()
    save(fig, "label_dist", out_dir)
    print("saved: label_dist.{png,pdf}")


def make_hedge_prev(out_dir):
    """Grouped bar chart: fraction of indications with >=1 hedge cue,
    by pathology cohort and label class."""
    neg_h = [15.1, 15.4, 17.0, 34.5]
    unc_h = [14.4, 12.5, 12.1, 15.5]
    pos_h = [11.2, 11.8, 11.0, 18.9]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for xi, a, b, c in zip(X, neg_h, unc_h, pos_h):
        for bx, val, col in [(xi - W, a, RUST), (xi, b, MUSTARD), (xi + W, c, GREEN)]:
            ax.bar(bx, val, width=W, color=col, edgecolor=INK, linewidth=1.1, zorder=3)
            ax.text(bx, val + 0.6, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK, zorder=4)

    ax.set_xticks(X)
    ax.set_xticklabels(GROUPS, fontsize=12.5)
    ax.set_ylim(0, 40)
    ax.set_ylabel("share of indications with $\\geq$1 hedge cue (%)",
                  fontsize=13.5, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    paper_axes(ax)
    ax.legend(handles=_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, fontsize=13, frameon=False, columnspacing=1.4, handletextpad=0.5)

    plt.tight_layout()
    save(fig, "hedge_prev", out_dir)
    print("saved: hedge_prev.{png,pdf}")


def main():
    ap = argparse.ArgumentParser(description="Generate label-dist and hedge-prev figures.")
    ap.add_argument("--out", default="figs", help="output directory (default: figs/)")
    args = ap.parse_args()

    make_label_dist(args.out)
    make_hedge_prev(args.out)


if __name__ == "__main__":
    main()
