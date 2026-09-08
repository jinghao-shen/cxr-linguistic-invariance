"""Paper-style result charts: AUROC grouped bars, cell-B forest plot,
decision-flip butterfly, and epsilon (unwarranted shift) panel.

All numbers are baked in from the final analysis runs; the figures are
purely presentational — no parquet loading happens here. Output: four
figures (PNG + PDF each) in --out.

  python figures/make_paper_charts.py --out figs/
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# classic journal palette: navy / terracotta / gold
GREEN   = "#1F497D"    # navy   -- primary series
RUST    = "#C0504D"    # red    -- secondary series
MUSTARD = "#C9A227"    # gold   -- tertiary / highlight
INK     = "#1A1A1A"
GRID    = "#D9D9D9"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["text.color"] = INK
plt.rcParams["axes.edgecolor"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK
plt.rcParams["axes.linewidth"] = 1.4


def paper_axes(ax):
    """Box border on all 4 sides + inward ticks, classic journal look."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.4)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, length=5, width=1.2, labelsize=13)
    ax.set_axisbelow(True)


def save(fig, stem, out_dir):
    """Write PNG + PDF to out_dir (white background for paper figures)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}",
                    facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {stem}.{{png,pdf}}")


# ============================================================
# Figure A: M1 / M2 / M3 AUROC grouped bar chart
# ============================================================
def make_auroc(out_dir):
    groups = ["Pleural\neffusion", "Consolidation", "Pulmonary\nedema"]
    m1 = [0.915, 0.806, 0.785]
    m2 = [0.790, 0.714, 0.689]
    m3 = [0.919, 0.841, 0.783]

    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=300)
    x, w = [0, 1.3, 2.6], 0.32
    for xi, a, b, c in zip(x, m1, m2, m3):
        for bx, val, col in [(xi - w, a, GREEN), (xi, b, RUST), (xi + w, c, MUSTARD)]:
            ax.bar(bx, val, width=w, color=col, edgecolor=INK, linewidth=1.1, zorder=3)
            ax.text(bx, val + 0.015, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold", color=INK, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=13.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("positive-vs-rest AUROC", fontsize=14.5, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    paper_axes(ax)
    ax.legend(handles=[
        Patch(facecolor=GREEN,   edgecolor=INK, label="M1 (image)"),
        Patch(facecolor=RUST,    edgecolor=INK, label="M2 (text)"),
        Patch(facecolor=MUSTARD, edgecolor=INK, label="M3 (fusion)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01),
       ncol=3, fontsize=12.5, frameon=False, columnspacing=1.4, handletextpad=0.5)

    plt.tight_layout()
    save(fig, "result_auroc", out_dir)


# ============================================================
# Figure B: headline forest plot -- cell B gap by pathology
# ============================================================
def make_forest(out_dir):
    rows = [
        ("Pleural effusion", 0.0337, 0.0286, 0.0388, 1072),
        ("Consolidation",    0.0452, 0.0373, 0.0535,  425),
        ("Pulmonary edema",  0.0255, 0.0181, 0.0325,  589),
    ]
    pooled = (0.0297, 0.0267, 0.0327, 3218)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=300)
    row_gap = 1.35
    ypos = [i * row_gap for i in range(len(rows), 0, -1)]
    for y, (name, gap, lo, hi, n) in zip(ypos, rows):
        ax.plot([lo, hi], [y, y], color=GREEN, lw=2.6, solid_capstyle="round", zorder=2)
        ax.plot(gap, y, "o", color=GREEN, ms=13, zorder=3,
                markeredgecolor=INK, markeredgewidth=1.1)
        ax.text(hi + 0.0028, y, f"{gap:+.3f}", va="center", ha="left",
                fontsize=13.5, color=INK, fontweight="bold")
        ax.text(-0.014, y, f"{name}\n" + r"$n$=" + f"{n:,}",
                va="center", ha="left", fontsize=12.5, color=INK)

    gap, lo, hi, n = pooled
    ax.plot([lo, hi], [0, 0], color=MUSTARD, lw=3.4, solid_capstyle="round", zorder=2)
    ax.plot(gap, 0, "D", color=MUSTARD, ms=13, zorder=3,
            markeredgecolor=INK, markeredgewidth=1.1)
    ax.text(hi + 0.0028, 0, f"{gap:+.4f}", va="center", ha="left",
            fontsize=13.5, color=INK, fontweight="bold")
    ax.text(-0.014, 0, f"Pooled (1 cue)\n" + r"$n$=" + f"{n:,}",
            va="center", ha="left", fontsize=12.5, color=INK, fontweight="bold")

    ax.axvline(0, color=INK, lw=1.2, zorder=1, linestyle=(0, (4, 3)))
    ax.set_xlim(-0.026, 0.068)
    ax.set_ylim(-0.9 * row_gap, (len(rows) + 0.9) * row_gap - (row_gap - 1))
    ax.set_yticks([])
    ax.set_xlabel("shift in predicted probability of the target pathology\n"
                  "(hedge-mask gap over placebo, 95% CI)",
                  fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    paper_axes(ax)
    ax.tick_params(axis="y", left=False, right=False)

    plt.tight_layout()
    save(fig, "result_forest", out_dir)


# ============================================================
# Figure C: decision flips -- hedge vs placebo (butterfly)
# ============================================================
def make_flips(out_dir):
    # (pathology, edit-type, up-count, down-count, net-label)
    groups2 = [
        ("Pleural\neffusion", "hedge mask", 32, 6,  "+2.4%"),
        ("Pleural\neffusion", "placebo",    13, 15, "-0.2%"),
        ("Consolidation",     "hedge mask", 17, 1,  "+3.7%"),
        ("Consolidation",     "placebo",    4,  9,  "-1.2%"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=300)
    bar_h, row_gap2 = 0.62, 1.35
    ys = [i * row_gap2 for i in range(len(groups2), 0, -1)]
    for y, (path, edit, up, down, net) in zip(ys, groups2):
        color = GREEN if edit == "hedge mask" else RUST
        alpha = 1.0 if edit == "hedge mask" else 0.75
        # right bar: net-positive flips; left bar: net-negative flips
        ax.barh(y, up,    height=bar_h, left=0,  color=color, edgecolor=INK,
                linewidth=1.0, alpha=alpha,        zorder=2)
        ax.barh(y, -down, height=bar_h, left=0,  color=color, edgecolor=INK,
                linewidth=1.0, alpha=alpha * 0.55, zorder=2)
        ax.text(up + 1.0,   y, f"{up}",   va="center", ha="left",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(-down - 1.0, y, f"{down}", va="center", ha="right",
                fontsize=13, color=INK)
        ax.text(38, y, f"net {net}", va="center", ha="left", fontsize=13, color=INK,
                fontweight="bold" if edit == "hedge mask" else "normal")
        label = f"{path}\n{edit}" if edit == "hedge mask" else edit
        ax.text(-26, y, label, va="center", ha="right", fontsize=12.5, color=INK)

    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    ax.set_xlim(-32, 58)
    ax.set_ylim(0.3 * row_gap2, (len(groups2) + 0.7) * row_gap2 - (row_gap2 - 1))
    ax.set_yticks([])
    ax.set_xlabel(
        r"studies crossing the decision threshold  ($\leftarrow$ to negative   |   to positive $\rightarrow$)",
        fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    paper_axes(ax)
    ax.tick_params(axis="y", left=False, right=False)
    ax.legend(handles=[
        Patch(facecolor=GREEN, edgecolor=INK,             label="hedge mask"),
        Patch(facecolor=RUST,  edgecolor=INK, alpha=0.75, label="placebo"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01),
       ncol=2, fontsize=12.5, frameon=False, columnspacing=1.4, handletextpad=0.5)

    plt.tight_layout()
    save(fig, "result_flips", out_dir)


# ============================================================
# Figure D: epsilon -- justified vs model shift per class
# ============================================================
def make_epsilon(out_dir):
    classes    = ["negative", "uncertain", "positive"]
    justified  = [-0.0012, -0.0032,  0.0053]
    justified_ci = [(-0.0166, 0.0127), (-0.0174, 0.0084), (-0.0126, 0.0223)]
    model      = [-0.0181, -0.0239,  0.0420]
    model_ci   = [(-0.0224, -0.0147), (-0.0284, -0.0186), (0.0385, 0.0451)]
    eps        = [-0.0169, -0.0208,  0.0367]
    eps_ci     = [(-0.0310, -0.0013), (-0.0342, -0.0066), (0.0203, 0.0547)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=300,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    yc = [2, 1, 0]
    # left panel: justified (RUST) vs model (GREEN) side-by-side per class
    for y, cls, j, jci, m, mci in zip(yc, classes, justified, justified_ci, model, model_ci):
        axL.plot(jci, [y + 0.16, y + 0.16], color=RUST, lw=2.4, solid_capstyle="round", zorder=2)
        axL.plot(j, y + 0.16, "o", color=RUST, ms=11, zorder=3,
                 markeredgecolor=INK, markeredgewidth=1.0)
        axL.plot(mci, [y - 0.16, y - 0.16], color=GREEN, lw=2.4, solid_capstyle="round", zorder=2)
        axL.plot(m, y - 0.16, "o", color=GREEN, ms=11, zorder=3,
                 markeredgecolor=INK, markeredgewidth=1.0)

    axL.axvline(0, color=INK, lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    axL.set_yticks(yc)
    axL.set_yticklabels(classes, fontsize=13.5)
    axL.set_xlim(-0.035, 0.055)
    axL.set_ylim(-0.7, 2.7)
    axL.set_xlabel("shift in predicted probability\nwhen the hedge is masked",
                   fontsize=12.5, fontweight="bold")
    axL.set_title("what the model does, and what the data allow",
                  fontsize=13, fontweight="bold", pad=10)
    axL.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    paper_axes(axL)
    axL.tick_params(axis="y", left=False, right=False)
    axL.legend(handles=[
        Patch(facecolor=RUST,  edgecolor=INK, label="justified, given the image"),
        Patch(facecolor=GREEN, edgecolor=INK, label="the model's shift"),
    ], loc="lower right", fontsize=10.5, frameon=True,
       edgecolor=INK, framealpha=0.95, borderpad=0.5)

    # right panel: epsilon = model - justified
    for y, cls, e, eci in zip(yc, classes, eps, eps_ci):
        axR.plot(eci, [y, y], color=INK, lw=2.4, solid_capstyle="round", zorder=2)
        axR.plot(e, y, "o", color=MUSTARD, ms=13, zorder=3,
                 markeredgecolor=INK, markeredgewidth=1.2)
        axR.text(e, y + 0.24, f"{e:+.3f}", va="bottom", ha="center",
                 fontsize=12.5, fontweight="bold", color=INK)

    axR.axvline(0, color=INK, lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    axR.set_yticks([])
    axR.set_xlim(-0.045, 0.065)
    axR.set_ylim(-0.7, 2.7)
    axR.set_xlabel(r"$\varepsilon$ = model $-$ justified" + "\n(95% CI, patient-clustered)",
                   fontsize=12.5, fontweight="bold")
    axR.set_title("the unwarranted part", fontsize=13, fontweight="bold", pad=10)
    axR.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    paper_axes(axR)

    plt.tight_layout()
    save(fig, "epsilon", out_dir)


def main():
    ap = argparse.ArgumentParser(description="Generate paper result figures.")
    ap.add_argument("--out", default="figs", help="output directory (default: figs/)")
    args = ap.parse_args()

    make_auroc(args.out)
    make_forest(args.out)
    make_flips(args.out)
    make_epsilon(args.out)


if __name__ == "__main__":
    main()
