"""Two-branch architecture diagram: DenseNet-121 (image) + ClinicalBERT (text) → fusion MLP.

Composite figure: a real CXR photograph and pre-rendered sub-diagrams
(densenet_main.png, bert_stack.png, densenet_legend.png) are loaded from
--figs and composited in matplotlib axes using exact pixel-crop coordinates.
The output is written back to the same directory as final_architecture.{png,pdf}.

Coordinate notes
----------------
The DenseNet sub-image is drawn in a canvas coordinate system (ax.set_xlim 0–2250,
ylim 160–690); the helper a_pt() maps pixel coordinates in the original PNG
to that canvas space.  phiI and phiT are right-edge–aligned (FEATURES_RIGHT)
so the two arrows converging into the concat box travel identical distances,
giving the diagram a symmetric look even though the two branches have very
different horizontal extents.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path

INK  = "#1A1A1A"
FILL = "#EFEAE0"   # light parchment — background for all boxes
GAP  = 55          # horizontal gap between every box-to-box arrow (canvas units)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})


def box(ax, x, y, w, h, text, fill=FILL, fontsize=12, fontweight="bold", lw=1.6):
    """Draw a sharp-cornered filled rectangle with centred text; return its bounding tuple."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=INK,
                            facecolor=fill, zorder=5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=INK,
            zorder=6, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, lw=2.2, connectionstyle=None, shrinkA=2, shrinkB=2):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17, linewidth=lw,
                        color=INK, zorder=6, connectionstyle=connectionstyle,
                        shrinkA=shrinkA, shrinkB=shrinkB)
    ax.add_patch(a)


def right_c(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left_c(b):
    x, y, w, h = b
    return (x, y + h / 2)


def make_architecture(figs_dir, out_dir):
    fig, ax = plt.subplots(figsize=(18, 4.6), dpi=300)
    ax.set_xlim(0, 2250)
    ax.set_ylim(160, 690)
    ax.axis("off")
    ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # ROW A: DenseNet-121 branch (top)
    # ------------------------------------------------------------------
    rowA_y0, rowA_y1 = 500, 669
    rowA_cy = (rowA_y0 + rowA_y1) / 2

    # Real CXR photo (90,016-study effusion cohort, positive label)
    xr_w, xr_h = 130, 130 / (390 / 320)   # preserve original aspect ratio
    xr_x0, xr_x1 = 20, 20 + xr_w
    xray = mpimg.imread(figs_dir / "example_edema_pos.jpg")
    ax.imshow(xray, extent=(xr_x0, xr_x1, rowA_cy - xr_h / 2, rowA_cy + xr_h / 2),
              cmap="gray", zorder=3, aspect="auto")
    ax.add_patch(Rectangle((xr_x0, rowA_cy - xr_h / 2), xr_w, xr_h,
                            fill=False, edgecolor=INK, linewidth=1.3, zorder=4))
    ax.text((xr_x0 + xr_x1) / 2, rowA_cy - xr_h / 2 - 10,
            "radiograph", ha="center", va="top", fontsize=11, color=INK, zorder=4)
    ax.text((xr_x0 + xr_x1) / 2, rowA_cy - xr_h / 2 - 32,
            "(real study, effusion cohort)", ha="center", va="top",
            fontsize=8.5, color="#6B6A5F", zorder=4)

    preproc = box(ax, xr_x1 + GAP, rowA_cy - 38, 145, 76,
                  "resize to 224²\n+ normalize", fontsize=9.5, fontweight="normal")
    arrow(ax, (xr_x1, rowA_cy), left_c(preproc))

    # Pre-rendered DenseNet block diagram (PNG); the original image's "Input"
    # stack is hidden below by a white Rectangle (see ix0/ix1/iy0/iy1 below).
    rowA_x0 = preproc[0] + preproc[2] + GAP
    rowA_x1 = rowA_x0 + 1000
    crop_w, crop_h = 2278, 385   # pixel dimensions of the source PNG
    main = mpimg.imread(figs_dir / "densenet_main.png")
    ax.imshow(main, extent=(rowA_x0, rowA_x1, rowA_y0, rowA_y1), zorder=1)
    arrow(ax, right_c(preproc), (rowA_x0, rowA_cy))

    def a_pt(ix, iy):
        """Map pixel coords (ix, iy) in the source PNG to canvas coords."""
        px = rowA_x0 + ix / crop_w * (rowA_x1 - rowA_x0)
        py = rowA_y1 - iy / crop_h * (rowA_y1 - rowA_y0)
        return px, py

    # Cover the original "Input" pink stack baked into the PNG (pixels 0–313 wide)
    # so our pre-processing box is the only input element in that region.
    ix0, iy0, ix1, iy1 = a_pt(-5, -5)[0], a_pt(-5, -5)[1], a_pt(313, 345)[0], a_pt(313, 345)[1]
    ax.add_patch(Rectangle((ix0, iy1), ix1 - ix0, iy0 - iy1,
                            facecolor="white", edgecolor="none", zorder=2))

    phiI_w, phiI_h = 230, 78
    phiI = box(ax, rowA_x1 + GAP, rowA_cy - phiI_h / 2, phiI_w, phiI_h,
               "image features\n" + r"$\in\mathbb{R}^{1024}$", fontsize=12)
    arrow(ax, (rowA_x1 - 3, rowA_cy), (rowA_x1 + GAP, rowA_cy))
    ax.text((rowA_x0 + rowA_x1) / 2, rowA_y1 + 12,
            "DenseNet-121 (image branch)",
            ha="center", fontsize=13, fontweight="bold", color=INK)

    # Shared right edge so both feature boxes' right sides align, making the
    # two arrows into concat the same length → symmetric merge point.
    FEATURES_RIGHT = phiI[0] + phiI_w

    # ------------------------------------------------------------------
    # ROW B: ClinicalBERT branch (bottom)
    # ------------------------------------------------------------------
    rowB_y0, rowB_y1 = 190, 430
    rowB_cy = (rowB_y0 + rowB_y1) / 2

    # Example indication text box
    quote_w = 190
    ax.add_patch(Rectangle((20, rowB_cy - 35), quote_w, 70,
                            facecolor="#FAF6ED", edgecolor=INK, linewidth=1.2, zorder=4))
    ax.text(20 + quote_w / 2, rowB_cy,
            "“88F SOB, eval for\npulmonary edema”",
            ha="center", va="center", fontsize=8.3, style="italic", color=INK, zorder=5)
    ax.text(20 + quote_w / 2, rowB_cy - 50, "indication",
            ha="center", va="top", fontsize=10.5, color=INK)

    embed = box(ax, 20 + quote_w + GAP, rowB_cy - 38, 190, 76,
                "tokenize + embed\nthe indication text",
                fontsize=9.5, fontweight="normal")
    arrow(ax, (20 + quote_w, rowB_cy), left_c(embed))

    # Pre-rendered BERT stack diagram
    bert_w_px, bert_h_px = 1194, 926
    bert_disp_w = 340
    bert_disp_h = bert_disp_w * bert_h_px / bert_w_px
    bert_x0 = embed[0] + embed[2] + GAP
    bert_x1 = bert_x0 + bert_disp_w
    bert_y0 = rowB_cy - bert_disp_h / 2
    bert_y1 = bert_y0 + bert_disp_h
    bert_img = mpimg.imread(figs_dir / "bert_stack.png")
    ax.imshow(bert_img, extent=(bert_x0, bert_x1, bert_y0, bert_y1), zorder=3)
    arrow(ax, right_c(embed), (bert_x0, rowB_cy))

    clspool = box(ax, bert_x1 + GAP, rowB_cy - 38, 190, 76,
                  "pool into one vector\nper indication",
                  fontsize=9.5, fontweight="normal")
    arrow(ax, (bert_x1, rowB_cy), left_c(clspool))

    # phiT is placed so its right edge equals phiI's right edge (FEATURES_RIGHT)
    phiT_w, phiT_h = 230, 78
    phiT = box(ax, FEATURES_RIGHT - phiT_w, rowB_cy - phiT_h / 2, phiT_w, phiT_h,
               "text features\n" + r"$\in\mathbb{R}^{768}$", fontsize=12)
    # Slight arc absorbs the longer horizontal stretch on the text branch
    arrow(ax, right_c(clspool), left_c(phiT), connectionstyle="arc3,rad=-0.08")

    ax.text((bert_x0 + bert_x1) / 2, bert_y1 + 20,
            "ClinicalBERT (text branch)  " + r"$\times$12 layers",
            ha="center", fontsize=13, fontweight="bold", color=INK)

    # ------------------------------------------------------------------
    # CONCAT + MLP head — centred between the two branch rows
    # ------------------------------------------------------------------
    mid_cy = (rowA_cy + rowB_cy) / 2
    concat_x0 = FEATURES_RIGHT + GAP
    concat = box(ax, concat_x0, mid_cy - 45, 160, 90,
                 "concat\n" + r"$\in\mathbb{R}^{1792}$", fontsize=12)
    # Curved arrows so both paths arrive at distinct y positions on the concat box
    arrow(ax, right_c(phiI), (concat_x0, mid_cy + 22), connectionstyle="arc3,rad=-0.2")
    arrow(ax, right_c(phiT), (concat_x0, mid_cy - 22), connectionstyle="arc3,rad=0.2")

    head_x0 = concat_x0 + 160 + GAP
    head = box(ax, head_x0, mid_cy - 50, 270, 100,
               "MLP + softmax\n{neg, unc, pos}", fontsize=12)
    arrow(ax, right_c(concat), left_c(head))

    # ------------------------------------------------------------------
    # DenseNet legend (block types) — top-right corner, no border
    # ------------------------------------------------------------------
    leg = mpimg.imread(figs_dir / "densenet_legend.png")
    leg_w_px, leg_h_px = 1780, 348
    leg_disp_w = 650
    leg_disp_h = leg_disp_w * leg_h_px / leg_w_px
    leg_x1, leg_y1 = 2250, 690
    leg_x0, leg_y0 = leg_x1 - leg_disp_w, leg_y1 - leg_disp_h
    ax.imshow(leg, extent=(leg_x0, leg_x1, leg_y0, leg_y1), zorder=3)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"final_architecture.{ext}",
                    dpi=300, facecolor="white", bbox_inches="tight")
    print(f"saved {out_dir}/final_architecture.{{png,pdf}}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Render the two-branch architecture diagram."
    )
    parser.add_argument(
        "--figs", type=Path, default=repo_root / "figs",
        help="Directory containing component images (densenet_main.png, bert_stack.png, "
             "densenet_legend.png, example_edema_pos.jpg) (default: <repo>/figs/)"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: same as --figs)"
    )
    args = parser.parse_args()
    out_dir = args.out if args.out is not None else args.figs
    out_dir.mkdir(parents=True, exist_ok=True)

    make_architecture(args.figs, out_dir)


if __name__ == "__main__":
    main()
