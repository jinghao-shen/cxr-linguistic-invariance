"""Print cell A / B / C counts and representative examples for each cohort.

Cell taxonomy (mirrors pipeline/analysis/placebo_contrast.py):
  A — hedged clause names the *target* pathology (legitimate signal)
  B — hedged clause names a *different* disease (the decisive test)
  C — hedged clause names *no* disease at all

Also reports word-count statistics and short (<= 9-word) canonical examples
for each cell, which were used to populate the paper's Table 1.

Usage:
    python scripts/compute_table.py [--cohort-dir pipeline/cohorts]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Disease regex patterns
# --------------------------------------------------------------------------
# Per-pathology pattern: matches indications *about* that disease
TARGET = {
    "edema":             r"\b(?:edema|chf|congestive heart failure|fluid overload|volume overload)",
    "pleural effusion":  r"\b(?:effusion|pleural fluid|hemothorax)",
    "consolidation":     r"\b(?:consolidat|infiltrat)",
}

# Union of all known disease terms — anything that names *a* disease
# (used to decide whether an indication is cell B vs. cell C)
ANY_DISEASE = re.compile(
    r"\b(?:edema|chf|congestive heart failure|fluid overload|volume overload|"
    r"pneumonia|pna|pneumothorax|ptx|effusion|infiltrat|consolidat|"
    r"atelectasis|embol|\bpe\b|nodule|mass|tumou?r|malignan|metasta|"
    r"fracture|abscess|tuberculo|hemothorax|aspiration|cardiomegaly)",
    re.IGNORECASE
)


def assign_cells(texts: pd.Series, target_pattern: str) -> np.ndarray:
    """Assign each hedged indication to cell A, B, or C.

    Vectorised: ~15× faster than row-wise apply on 90k rows.
    """
    tgt = texts.str.contains(target_pattern, case=False, regex=True)
    any_dis = texts.str.contains(ANY_DISEASE.pattern, flags=re.IGNORECASE, regex=True)
    return np.where(tgt, "A", np.where(any_dis, "B", "C"))


def report_cohort(name: str, parquet_path: Path, target_key: str) -> pd.DataFrame:
    """Print counts and examples for one cohort; return the annotated hedged DataFrame."""
    df = pd.read_parquet(parquet_path)
    hedged = df[df["has_hedge"].astype(bool)].copy()
    hedged["cell"] = assign_cells(hedged["h"].astype(str), TARGET[target_key])

    counts = hedged["cell"].value_counts().to_dict()
    print(f"\n{'='*60}")
    print(f"{name}  (total={len(df):,}, hedged={len(hedged):,})")
    print(f"{'='*60}")
    for c in ["A", "B", "C"]:
        n = counts.get(c, 0)
        ex = hedged.loc[hedged["cell"] == c, "h"].iloc[0] if n > 0 else None
        print(f"  cell {c}  n={n:,}  example={ex!r}")

    return hedged


def short_examples(hedged: pd.DataFrame, name: str, max_words: int = 9) -> None:
    """Print the two shortest indications for each cell (≤ max_words words).

    Short examples are most legible when typeset in a paper table.
    """
    hedged = hedged.copy()
    hedged["wc"] = hedged["h"].astype(str).str.split().str.len()
    print(f"\n--- {name}: short examples (≤{max_words} words) ---")
    for c in ["A", "B", "C"]:
        sub = hedged[(hedged["cell"] == c) & (hedged["wc"] <= max_words)]
        if sub.empty:
            print(f"  {c}: (none within word limit)")
            continue
        for row in sub["h"].head(2):
            print(f"  {c}: {row!r}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Print cell-count table for all cohorts.")
    parser.add_argument(
        "--cohort-dir", type=Path,
        default=repo_root / "pipeline" / "cohorts",
        help="Directory containing *_ext.parquet cohort files (default: <repo>/pipeline/cohorts/)"
    )
    args = parser.parse_args()

    cohorts = {
        "Pleural effusion": (args.cohort_dir / "pleural_effusion_ext.parquet", "pleural effusion"),
        "Pulmonary edema":  (args.cohort_dir / "edema_ext.parquet",            "edema"),
        "Consolidation":    (args.cohort_dir / "consolidation_ext.parquet",    "consolidation"),
    }

    hedged_frames = {}
    for name, (path, target_key) in cohorts.items():
        hedged_frames[name] = report_cohort(name, path, target_key)

    # Word-count statistics on the pleural effusion cohort (primary / largest)
    df_effusion = pd.read_parquet(cohorts["Pleural effusion"][0])
    wc = df_effusion["h"].astype(str).str.split().str.len()
    print(f"\nWord count stats (effusion cohort, all rows with indication):")
    print(f"  median={wc.median():.0f}  min={wc.min()}  max={wc.max()}")

    for name, hedged in hedged_frames.items():
        short_examples(hedged, name)

    # Spot-check: consolidation cell A where the literal string 'consolidat' appears
    hedged_cons = hedged_frames["Consolidation"].copy()
    hedged_cons["wc"] = hedged_cons["h"].astype(str).str.split().str.len()
    sub = hedged_cons[
        (hedged_cons["cell"] == "A")
        & (hedged_cons["h"].str.contains("consolidat", case=False))
        & (hedged_cons["wc"] <= 10)
    ]
    print(f"\n--- Consolidation cell A with literal 'consolidat' (≤10 words): {len(sub)} matches ---")
    for x in sub["h"].head(8):
        print(f"  - {x!r}")


if __name__ == "__main__":
    main()
