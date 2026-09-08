"""How much *should* masking the hedge move the prediction, given the image?

The model shifts p_pos by some amount when c1 removes the epistemic frame. Calling
that shift right or wrong needs a benchmark, and the naive one is badly confounded:
hedged indications mark a different kind of study (outpatient work-up vs inpatient
follow-up), and the radiograph already shows that -- tubes, lines, positioning. So
the marginal association between hedging and the label mostly reflects something
the image branch can see anyway.

The benchmark here is the ATT of removing the hedge, holding the image fixed:

    justified = E[ Y | do(no hedge), img ] - E[ Y | hedge, img ]   over hedged rows

estimated by G-computation -- logistic regression of Y on the hedge flag plus a
spline in the image-only model's logit, then predicting each hedged row both ways
and averaging the difference. The image-only score is the sufficient statistic we
have for "what the radiograph shows"; a spline rather than quintile bins because
binning threw away most of the precision (CI width 0.055 -> see --bins to compare).

Reported next to the model's own shift on the same rows, with the difference and a
patient-clustered CI: that difference, not the raw shift, is the claim.

  conda run -n base python -m pipeline.justified_shift \
      --pathology pleural_effusion --cell B
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from pipeline.data import DISEASE_RE

ROOT = Path(__file__).resolve().parent.parent
TARGET_RE = {
    "edema": r"\b(?:edema|chf|congestive heart failure|fluid overload|volume overload)",
    "pleural_effusion": r"\b(?:effusion|pleural fluid|hemothorax)",
    "consolidation": r"\b(?:consolidat|infiltrat)",
    "pneumonia": r"\b(?:pneumonia|pna|infiltrat|consolidat|infection|septic)",
}


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def att_gcomp(df, df_spline=5, seed=0, cols=("img",), ycol="y"):
    """ATT of removing the hedge, standardising over the image representation.

    `cols` is the image summary to adjust on. p_pos alone is a 1-D projection and
    can leave residual confounding: an ICU film shows tubes that predict both
    hedging and the finding beyond the finding's own probability. Passing all
    three class scores adjusts on the image branch's full output simplex.
    """
    x = np.column_stack([_logit(df[c].values) for c in cols])
    st = SplineTransformer(n_knots=df_spline, degree=3, include_bias=False)
    B = st.fit_transform(x)
    hed = df["hed"].values.astype(float).reshape(-1, 1)
    X = np.hstack([hed, B])
    y = df[ycol].values
    if y.min() == y.max():
        return np.nan
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    treated = df["hed"].values.astype(bool)
    if treated.sum() == 0:
        return np.nan
    Bt = B[treated]
    p_hedged = clf.predict_proba(np.hstack([np.ones((Bt.shape[0], 1)), Bt]))[:, 1]
    p_plain = clf.predict_proba(np.hstack([np.zeros((Bt.shape[0], 1)), Bt]))[:, 1]
    return float((p_plain - p_hedged).mean())


def att_bins(df, nq=5):
    """the earlier quintile estimator, kept so the precision gain is visible"""
    try:
        q = pd.qcut(df["img"], nq, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    num = den = 0.0
    for _, g in df.groupby(q):
        h, n = g[g.hed], g[~g.hed]
        if len(h) < 5 or len(n) < 5:
            continue
        num += len(g) * (n.y.mean() - h.y.mean())
        den += len(g)
    return num / den if den else np.nan


CLASSES = [("neg", 0), ("unc", -1), ("pos", 1)]     # name -> CheXbert label value


def load(pathology, cell):
    coh = pd.read_parquet(ROOT / "pipeline" / "cohorts" / f"{pathology}_ext.parquet")
    m1 = pd.read_parquet(ROOT / "runs" / f"{pathology}_m1" / "preds_test.parquet")
    m3 = pd.read_parquet(ROOT / "runs" / f"{pathology}_m3" / "preds_test.parquet")
    im = m1[["path_to_image", "p_pos_h", "p_neg_h", "p_unc_h"]].rename(
        columns={"p_pos_h": "img", "p_neg_h": "img_neg", "p_unc_h": "img_unc"})
    d = m3.merge(im, on="path_to_image").merge(
        coh[["path_to_image", "h", "deid_patient_id"]], on="path_to_image")
    t = d["h"].astype(str)
    is_a = t.str.contains(TARGET_RE[pathology], case=False, regex=True)
    is_b = t.str.contains(DISEASE_RE) & ~is_a
    d["cell"] = np.where(is_a, "A", np.where(is_b, "B", "C"))
    d = d.copy()
    # One-vs-rest per class rather than dropping the uncertain rows: pos-vs-rest
    # alone hides the class the hedging is supposed to act on, and keeping the
    # uncertain rows is also free sample size (40% of the consolidation cohort).
    d["hed"] = d["has_hedge"].astype(bool)
    for name, val in CLASSES:
        d[f"y_{name}"] = (d["label"] == val).astype(int)
        d[f"mshift_{name}"] = d[f"p_{name}_c1"] - d[f"p_{name}_h"]
    d["y"] = d["y_pos"]                             # default outcome
    d["mshift"] = d["mshift_pos"]
    return d[d["cell"] == cell] if cell != "all" else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathology", default="pleural_effusion")
    ap.add_argument("--cell", default="B", choices=["A", "B", "C", "all"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--bins", action="store_true", help="also show the quintile estimator")
    ap.add_argument("--rich", action="store_true",
                    help="adjust on all three M1 class scores, not just p_pos")
    args = ap.parse_args()

    d = load(args.pathology, args.cell)
    hed = d[d["hed"]]
    print(f"\n=== {args.pathology}, cell {args.cell}: is the hedge response calibrated? ===")
    print(f"  n={len(d)} (hedged {len(hed)}, un-hedged {len(d)-len(hed)}), "
          f"patients={d.deid_patient_id.nunique()}")
    for name, val in CLASSES:
        print(f"    {name}: {int((d.label == val).sum())}", end="")
    print()
    COLS = ("img", "img_neg", "img_unc") if args.rich else ("img",)

    groups = list(d.groupby("deid_patient_id", sort=False).indices.values())
    print(f"\n  {'class':6s} {'justified':>10s} {'95% CI':>19s} {'model':>9s} "
          f"{'95% CI':>19s} {'model-just':>11s} {'95% CI':>19s}  verdict")
    for name, _ in CLASSES:
        yc, mc = f"y_{name}", f"mshift_{name}"
        obs_j = att_gcomp(d, cols=COLS, ycol=yc)
        obs_m = float(hed[mc].mean())
        rng = np.random.default_rng(0)
        J, M, D = [], [], []
        for _ in range(args.n_boot):
            s_ = d.iloc[np.concatenate(
                [groups[i] for i in rng.integers(0, len(groups), len(groups))])]
            j = att_gcomp(s_, cols=COLS, ycol=yc)
            if np.isnan(j):
                continue
            sh = s_[s_["hed"]]
            if len(sh) < 5:
                continue
            J.append(j); M.append(sh[mc].mean()); D.append(sh[mc].mean() - j)
        if not D:
            print(f"  {name:6s}  (degenerate)")
            continue
        ci = lambda a: f"[{np.percentile(a,2.5):+.4f},{np.percentile(a,97.5):+.4f}]"
        lo, hi = np.percentile(D, 2.5), np.percentile(D, 97.5)
        verdict = ("MORE than justified" if lo > 0 else
                   "LESS than justified" if hi < 0 else "n.s.")
        print(f"  {name:6s} {obs_j:+10.4f} {ci(J):>19s} {obs_m:+9.4f} {ci(M):>19s} "
              f"{obs_m-obs_j:+11.4f} {ci(D):>19s}  {verdict}")
    print("\n  justified = image-adjusted ATT of removing the hedge on P(class);")
    print("  model = mean shift in the model's probability for that class.")


if __name__ == "__main__":
    main()
