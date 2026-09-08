"""The decisive test: is the model's hedge response hedge-SPECIFIC?

Masking a hedge that is about ANOTHER disease moves the model even though that
cue carries no information about the target pathology. That is only a shortcut if
the shift is specific to hedges. The placebo view masks a matched number/length of
*neutral* words (never a cue, never a pathology mention), so:

    delta(c1) >> delta(placebo)  -> hedge-specific, the finding holds
    delta(c1) ~= delta(placebo)  -> plain perturbation sensitivity, the line ends

Cells: A = the hedge is about the target disease (legitimate signal),
       B = the hedge is about a DIFFERENT disease (no information about the
           target, so any response is unjustifiable -- this is the test),
       C = hedged but no disease named at all.
Restricted to `placebo_ok` rows, where BOTH operators actually alter the text;
short indications ("r/o edema") leave no neutral word to mask and would flatter
the hedge if included. CIs are patient-clustered bootstrap.

  conda run -n base python -m pipeline.placebo_contrast \
      --m3 runs/edema_m3 --cohort pipeline/cohorts/edema_ext.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Target-pathology synonyms. The cohort's `pathology` column selects the entry.
TARGET = {
    "edema": r"\b(?:edema|chf|congestive heart failure|fluid overload|volume overload)",
    "pneumonia": r"\b(?:pneumonia|pna|infiltrat|consolidat|infection|septic)",
    "pleural effusion": r"\b(?:effusion|pleural fluid|hemothorax)",
    "consolidation": r"\b(?:consolidat|infiltrat)",
}
CLASSES = ["neg", "unc", "pos"]

ANY_DISEASE = (r"\b(?:edema|chf|congestive heart failure|fluid overload|volume overload|"
               r"pneumonia|pna|pneumothorax|ptx|effusion|infiltrat|consolidat|"
               r"atelectasis|embol|\bpe\b|nodule|mass|tumou?r|malignan|metasta|"
               r"fracture|abscess|tuberculo|hemothorax|aspiration|cardiomegaly)")


def clustered_boot(df, col, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = list(df.groupby("patient", sort=False).indices.values())
    n = len(groups)
    vals = df[col].values
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        out[b] = vals[np.concatenate([groups[i] for i in pick])].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    pr = pd.read_parquet(Path(args.m3) / "preds_test.parquet")
    if "p_pos_p" not in pr:
        raise SystemExit(
            "this run has no placebo view (p_pos_p). It predates the 2026-07-29 "
            "train.py; the contrast cannot be computed from it.")
    coh = pd.read_parquet(args.cohort)
    path = coh["pathology"].iloc[0].lower()
    tgt = TARGET.get(path)
    if tgt is None:
        raise SystemExit(f"no synonym list for pathology {path!r}")

    d = pr.merge(coh[["path_to_image", "h", "placebo_ok", "deid_patient_id"]],
                 on="path_to_image").rename(columns={"deid_patient_id": "patient"})
    t = d["h"].astype(str)
    d["cell"] = np.where(t.str.contains(tgt, case=False, regex=True),
                         f"A: about {path.upper()}",
                 np.where(t.str.contains(ANY_DISEASE, case=False, regex=True),
                          "B: about OTHER disease", "C: no disease named"))
    # Per class, not pos-vs-rest only. The hedge's linguistic job is to hold the
    # reader at "uncertain", so the uncertain class is where the effect should be
    # most interpretable -- and pooling it into "rest" hides exactly that.
    for name in CLASSES:
        d[f"d_c1_{name}"] = d[f"p_{name}_c1"] - d[f"p_{name}_h"]
        d[f"d_pl_{name}"] = d[f"p_{name}_p"] - d[f"p_{name}_h"]
        d[f"gap_{name}"] = d[f"d_c1_{name}"] - d[f"d_pl_{name}"]
    d["d_c1"], d["d_pl"] = d["d_c1_pos"], d["d_pl_pos"]
    d["d_gap"] = d["gap_pos"]

    elig = d[d["placebo_ok"].astype(bool)]
    print(f"\n=== placebo contrast: {path}, placebo_ok rows only "
          f"(n={len(elig)} of {int(d.placebo_ok.notna().sum())} test rows) ===")
    print(f"  {'cell':24s} {'class':5s} {'n':>5s} {'d(c1)':>9s} {'d(placebo)':>11s} "
          f"{'gap':>9s} {'95% CI of gap':>20s}  hedge-specific?")
    for c in sorted(elig["cell"].unique()):
        s = elig[elig["cell"] == c]
        if len(s) < 20:
            print(f"  {c:24s} {'':5s} {len(s):5d}   (too few)")
            continue
        for name in CLASSES:
            lo, hi = clustered_boot(s, f"gap_{name}", args.n_boot)
            spec = "yes" if lo > 0 or hi < 0 else "no"
            print(f"  {c:24s} {name:5s} {len(s):5d} {s[f'd_c1_{name}'].mean():+9.4f} "
                  f"{s[f'd_pl_{name}'].mean():+11.4f} {s[f'gap_{name}'].mean():+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  {spec}")

    print("\n  gap = delta(c1) - delta(placebo), the hedge-specific part.")
    print("  Cell B is the test: the cue carries no information about the target,")
    print("  so a gap whose CI excludes 0 there is an unjustifiable dependency.")

    # Decision-level restatement, cell B.
    #
    # Total flip rate is direction-blind and therefore the wrong instrument: a
    # perturbation that jitters predictions symmetrically crosses the threshold
    # often while biasing nothing. The claim is that the hedge shifts decisions
    # *systematically*, so report the two directions separately and their
    # asymmetry (net = up - down). A noise control should come out near zero net
    # however much churn it produces.
    b = elig[elig["cell"] == "B: about OTHER disease"]
    if len(b) >= 20:
        thr = float(np.quantile(d["p_pos_h"], 1 - (d["label"] == 1).mean()))
        print(f"\n  cell-B decision changes at the prevalence-matched threshold "
              f"{thr:.3f} (n={len(b)}):")
        print(f"    {'edit':14s} {'any flip':>10s} {'neg->pos':>9s} {'pos->neg':>9s} "
              f"{'net (up-down)':>15s} {'95% CI of net':>20s}")
        for col, tag in [("p_pos_c1", "hedge mask"), ("p_pos_p", "placebo mask")]:
            was, now = b["p_pos_h"] > thr, b[col] > thr
            up, down = (now & ~was), (was & ~now)
            bb = b.assign(net=(up.astype(float) - down.astype(float)))
            lo, hi = clustered_boot(bb, "net", args.n_boot)
            print(f"    {tag:14s} {int((up|down).sum()):10d} {int(up.sum()):9d} "
                  f"{int(down.sum()):9d} {bb.net.mean()*100:+14.1f}% "
                  f"[{lo*100:+.1f}%, {hi*100:+.1f}%]")
        print("    (any-flip counts churn; `net` is the systematic part -- the "
              "placebo may churn more while biasing nothing)")


if __name__ == "__main__":
    main()
