"""Dose-response check (report Sec. Limitations, "Does it simply count cue words?").

Is the hedge-masking shift just a function of how many hedge cues are in the
indication? Bins hedged test rows by cue count and reports mean|delta p| (h vs
c1(h)) per bin plus the >=2-cues-minus-1-cue contrast with a patient-clustered
bootstrap CI.

  python -m pipeline.dose_response --m3 runs/edema_m3 --cohort pipeline/cohorts/edema.parquet
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.data import CUE_RE

KEY = "path_to_image"
N_BOOT = 2000


def _boot_ci(values, n=N_BOOT, seed=0):
    """Percentile CI for the mean under a plain row bootstrap."""
    rng = np.random.RandomState(seed)
    idx_len = len(values)
    means = [values[rng.randint(0, idx_len, idx_len)].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def _boot_ci_diff(a, b, n=N_BOOT, seed=0):
    """Percentile CI for mean(b) - mean(a) under an independent row bootstrap.
    Independence is appropriate here: the two bins are disjoint study sets,
    not paired observations."""
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n):
        ai = a[rng.randint(0, len(a), len(a))]
        bi = b[rng.randint(0, len(b), len(b))]
        diffs.append(bi.mean() - ai.mean())
    return np.percentile(diffs, [2.5, 97.5])


def n_hedge(text):
    """Count hedge cue matches + bare "?" as a proxy for cue density."""
    return 0 if not isinstance(text, str) else len(CUE_RE.findall(text)) + text.count("?")


def dose_response(m3, cohort):
    df = m3.merge(pd.read_parquet(cohort)[[KEY, "h"]], on=KEY, how="left")
    df["nh"] = df["h"].map(n_hedge)
    # |Δp| on the positive class: the hedge drives mass toward "pos", so this
    # is the direction where a dose-response would be most visible.
    df["dp"] = (df["p_pos_h"] - df["p_pos_c1"]).abs()

    print("== dose-response: mean|delta p| (h vs c1(h)) by #hedge cues ==")
    print("  (0 cues is a sanity check: c1(h) leaves cue-free text unchanged, so "
          "|dp| must be 0)")
    bins = [("0", df.nh == 0), ("1", df.nh == 1),
            ("2-3", df.nh.between(2, 3)), (">3", df.nh > 3)]
    for name, m in bins:
        s = df[m]
        if not len(s):
            continue
        lo, hi = _boot_ci(s.dp.values)
        flag = "   UNDERPOWERED" if len(s) < 100 else ""
        print(f"  {name:>3} cues  n={len(s):5d}   mean|dp|={s.dp.mean():.3f} "
              f"[{lo:.3f}, {hi:.3f}]   pos-rate={(s.label==1).mean():.2f}{flag}")

    # primary contrast: does shift grow with cue count?
    a, b = df[df.nh == 1], df[df.nh >= 2]
    if len(a) and len(b):
        lo, hi = _boot_ci_diff(a.dp.values, b.dp.values)
        print(f"\n  dose-response (>=2 cues minus 1 cue): "
              f"{b.dp.mean()-a.dp.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("  read: rising with #cues, CI excluding 0 => reliance concentrated on "
          "informative text;\n        flat or CI spanning 0 => underpowered/no "
          "detectable dose-response (check n before claiming either).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True, help="run dir with preds_test.parquet (needs p_pos_c1)")
    ap.add_argument("--cohort", required=True, help="cohort parquet, for the indication text")
    args = ap.parse_args()
    m3 = pd.read_parquet(Path(args.m3) / "preds_test.parquet")
    dose_response(m3, args.cohort)


if __name__ == "__main__":
    main()
