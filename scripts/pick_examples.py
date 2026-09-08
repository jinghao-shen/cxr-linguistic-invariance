"""Find real studies to illustrate the example figure (fig:example) in the paper.

Reads the local cohorts (no images needed) and selects, from the TEST split:
  - edema:     label == 1 (positive)  AND has_hedge  -> "possible edema ..."
  - pneumonia: label == 0 (negative)  AND has_hedge  -> "rule out pneumonia ..."
so the picture, its indication, and its label are one real study and agree with
the figure caption. Prints each study's path_to_image, indication h, and label.

  python scripts/pick_examples.py --cohort-dir pipeline/cohorts
"""
import argparse
from pathlib import Path
import pandas as pd

LABELS = {1: "positive", 0: "negative", -1: "uncertain"}


def pick(cohort_dir, name, want_label):
    df = pd.read_parquet(Path(cohort_dir) / f"{name}.parquet")
    sel = df[(df.split == "test") & (df.label == want_label) & (df.has_hedge)]
    if sel.empty:  # fall back to any split if test is thin
        sel = df[(df.label == want_label) & (df.has_hedge)]
    # shortest indication first: cleanest to typeset in the figure
    sel = sel.assign(_len=sel.h.str.len()).sort_values("_len")
    for _, r in sel.head(5).iterrows():
        print(f"  path : {r.path_to_image}")
        print(f"  label: {LABELS[int(r.label)]}   split={r.split}")
        print(f"  h    : {r.h!r}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Print candidate example-figure studies.")
    ap.add_argument("--cohort-dir", default="pipeline/cohorts",
                    help="directory containing <name>.parquet cohort files "
                         "(default: pipeline/cohorts)")
    args = ap.parse_args()

    print("=== EDEMA (want positive, hedged) ===")
    pick(args.cohort_dir, "edema", 1)
    print("=== PNEUMONIA (want negative, hedged) ===")
    pick(args.cohort_dir, "pneumonia", 0)
    print("Pick one line from each block; those two `path`s are the images to fetch.")


if __name__ == "__main__":
    main()
