"""Data assembly for the hedge-shortcut CXR study.

Builds, per pathology, a cohort of (image path, clinical indication h, masked
indication c(h), 3-class label, patient split). No images are required at this
stage; the image branch consumes `path_to_image` later on a machine that has the
CheXpert Plus files.
"""
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "chexpert_main.parquet"
LABELS = ROOT / "impression_fixed.json"       # CheXbert labels on the impression section
OUT = ROOT / "pipeline" / "cohorts"

# --- hedge lexicon (frozen before training; see report Sec. 3.2) -------------
# Epistemic-frame cues of the *indication* register. Sources: CheXpert/NegBio
# uncertainty rules, Panicek & Hricak 2016, BioScope (Vincze 2008), plus a
# data-driven scan of section_clinical_history. Ordered longest-first so that
# multi-word frames are masked before their sub-tokens.
HEDGE_CUES = [
    "cannot be excluded", "cannot exclude", "cannot rule out", "please evaluate",
    "concerning for", "suggestive of", "suspicious for", "worrisome for",
    "question of", "questionable", "evaluate for", "assessment for", "assess for",
    "concern for", "check for", "eval for", "rule out", "may represent",
    "could be", "presumed", "probable", "possibly", "possible", "suspected",
    "suspect", "versus", "query", "r/o", "vs.", "vs", "likely",
]
# procedural / non-suspicion phrases explicitly NOT treated as hedges
# (documented for reproducibility; they are simply absent from HEDGE_CUES)
_MASK = "[MASK]"


def _cue_pattern():
    parts = []
    for c in HEDGE_CUES:
        esc = re.escape(c)
        # word-boundary for alphanumeric cues; r/o and vs. handled by escape
        parts.append(rf"\b{esc}" if c[0].isalpha() else esc)
    return re.compile("|".join(parts), flags=re.IGNORECASE)


CUE_RE = _cue_pattern()


def mask_hedge(text: str) -> str:
    """c(.) : replace the epistemic frame with [MASK], keep pathology names."""
    if not isinstance(text, str):
        return text
    out = CUE_RE.sub(_MASK, text)
    out = re.sub(r"\?", _MASK, out)                 # bare '?' is a hedge marker
    out = re.sub(r"(\s*\[MASK\]\s*)+", " [MASK] ", out)   # collapse repeats
    return out.strip()


def mask_phrase(text: str) -> str:
    """c2(.) : mask the whole hedged clause (frame + the pathology it governs),
    from each hedge cue to the next clause terminator {, . ; \\n} or end."""
    if not isinstance(text, str):
        return text
    out, i = [], 0
    for m in CUE_RE.finditer(text):
        if m.start() < i:
            continue
        out.append(text[i:m.start()])
        end = re.search(r"[,.;\n]", text[m.start():])
        stop = m.start() + (end.start() if end else len(text) - m.start())
        out.append(_MASK)
        i = stop
    out.append(text[i:])
    return re.sub(r"(\s*\[MASK\]\s*)+", " [MASK] ", "".join(out)).strip()


# Pathology mentions are excluded from placebo candidates: naming a disease
# carries signal of its own (measured separately via c2 / the naming contrast),
# so masking one would contaminate the control. The placebo therefore masks
# *uninformative* words, making it the right null for "is the hedge special".
DISEASE_RE = re.compile(
    r"\b(edema|chf|congestive heart failure|fluid overload|volume overload|"
    r"pneumonia|pna|pneumothorax|ptx|effusion|infiltrat\w*|consolidat\w*|"
    r"atelectasis|embol\w*|nodule|mass|tumou?r|malignan\w*|metasta\w*|"
    r"fracture|abscess|tuberculo\w*|hemothorax|aspiration|cardiomegaly)",
    flags=re.IGNORECASE)


def mask_placebo(text: str, seed: int = 0) -> str:
    """Placebo control for c(.): mask the same NUMBER of word-spans, of the same
    word-lengths, but drawn from NON-cue positions.

    Without this, a nonzero response to c(.) is ambiguous: inserting [MASK] at
    all perturbs the input, and any model shifts a little under any perturbation.
    The placebo matches c(.) on every surface property we can control (count of
    [MASK] tokens, span lengths, and identity on cue-free text) and differs only
    in *what* is masked, so the contrast delta(c) - delta(placebo) isolates the
    hedge. Deterministic given the text, so the view is stable across runs.
    """
    if not isinstance(text, str):
        return text
    words = text.split()
    if not words:
        return text
    # word indices covered by a cue, and the word-length of each cue span
    covered, spans = set(), []
    for m in CUE_RE.finditer(text):
        lo = len(text[:m.start()].split())
        hi = lo + max(1, len(m.group().split()))
        spans.append(hi - lo)
        covered.update(range(lo, hi))
    if "?" in text:
        spans.append(1)
    if not spans:
        return text                       # identity, exactly as c(.) is here
    for m in DISEASE_RE.finditer(text):   # never mask a pathology mention
        lo = len(text[:m.start()].split())
        covered.update(range(lo, lo + max(1, len(m.group().split()))))
    rng = np.random.default_rng(abs(hash(text)) % (2**32) + seed)
    out = list(words)
    taken = set(covered)
    for L in spans:
        starts = [i for i in range(len(words) - L + 1)
                  if not any(j in taken for j in range(i, i + L))]
        if not starts:
            break
        s = int(rng.choice(starts))
        for j in range(s, s + L):
            out[j] = _MASK
            taken.add(j)
    return re.sub(r"(\s*\[MASK\]\s*)+", " [MASK] ", " ".join(out)).strip()


def has_hedge(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(CUE_RE.search(text)) or "?" in text


_PLACEHOLDER = {"", "nan", "none", "None", "NONE"}


def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.mask(s.str.lower().isin({"", "nan", "none"}))


def patient_split(patient_ids: pd.Series, seed: int = 0,
                  fracs=(0.70, 0.15, 0.15)) -> pd.Series:
    """Deterministic 70/15/15 split at the patient level (disjoint patients)."""
    uniq = np.sort(patient_ids.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    assign = {}
    for i, pid in enumerate(uniq):
        assign[pid] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return patient_ids.map(assign)


def build_cohort(pathology: str, seed: int = 0,
                 include_history: bool = True) -> pd.DataFrame:
    """`include_history`: CheXpert Plus writes the pre-imaging indication under
    two mutually exclusive headers -- `section_clinical_history` (62.1% of
    studies) and `section_history` (17.1%); only 5 studies carry both. Taking
    the first non-empty of the two recovers ~17% of studies that an earlier
    clinical_history-only cohort silently dropped. Same register, same
    pre-imaging provenance, so no leakage boundary is crossed."""
    main = pd.read_parquet(MAIN)
    main["path_to_image"] = main["path_to_image"].astype(str).str.strip()
    lab = pd.read_json(LABELS, lines=True)
    lab["path_to_image"] = lab["path_to_image"].astype(str).str.strip()

    df = main[["path_to_image", "deid_patient_id", "frontal_lateral",
               "section_clinical_history", "section_history"]].merge(
        lab[["path_to_image", pathology]], on="path_to_image", how="inner")

    # frontal only
    df = df[df["frontal_lateral"].astype(str).str.lower().str.startswith("front")]
    # non-empty indication: clinical_history, falling back to history
    df["h"] = _clean_text(df["section_clinical_history"])
    if include_history:
        df["h_src"] = np.where(df["h"].notna(), "clinical_history", "history")
        df["h"] = df["h"].fillna(_clean_text(df["section_history"]))
    else:
        df["h_src"] = "clinical_history"
    df = df[df["h"].notna()]
    # keep 3 classes {0 neg, -1 unc, 1 pos}; drop blanks/NaN
    df = df[df[pathology].isin([0.0, -1.0, 1.0])].copy()
    df["label"] = df[pathology].astype(int)          # {-1,0,1}

    df["h_masked"] = df["h"].map(mask_hedge)        # c1: frame-only
    df["h_phrase"] = df["h"].map(mask_phrase)        # c2: whole hedged clause
    df["h_placebo"] = df["h"].map(mask_placebo)      # placebo: matched non-cue mask
    df["has_hedge"] = df["h"].map(has_hedge)
    # Short indications ("r/o edema") can have no maskable neutral word left once
    # cues and pathology mentions are excluded, so the placebo degenerates to the
    # identity there. Comparing delta(c) against a no-op placebo would flatter the
    # hedge, so flag the rows where BOTH operators actually alter the text; the
    # placebo contrast must be restricted to these.
    df["placebo_ok"] = df["has_hedge"] & (df["h_placebo"] != df["h"]) \
        & (df["h_masked"] != df["h"])
    df["split"] = patient_split(df["deid_patient_id"], seed=seed)
    df["pathology"] = pathology

    # sanity: patient disjointness across splits
    g = df.groupby("deid_patient_id")["split"].nunique()
    assert (g == 1).all(), "patient leak across splits!"
    return df[["path_to_image", "deid_patient_id", "pathology", "h", "h_src",
               "h_masked", "h_phrase", "h_placebo", "has_hedge", "placebo_ok", "label",
               "split"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathologies", nargs="+", default=["Pneumonia", "Edema"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-history", action="store_true",
                    help="clinical_history only (the original, smaller cohort)")
    ap.add_argument("--suffix", default="_ext",
                    help="filename suffix; '' overwrites the original cohorts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for p in args.pathologies:
        c = build_cohort(p, seed=args.seed, include_history=not args.no_history)
        # slug: CheXbert column names carry spaces ("Pleural Effusion"), which
        # would otherwise become filenames that need quoting everywhere downstream
        slug = re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")
        out = OUT / f"{slug}{'' if args.no_history else args.suffix}.parquet"
        c.to_parquet(out)
        # report
        print(f"\n=== {p}  (n={len(c)}, patients={c.deid_patient_id.nunique()}) -> {out.name}")
        print("  indication source: " +
              "  ".join(f"{k}={v}" for k, v in c.h_src.value_counts().items()))
        for sp in ["train", "val", "test"]:
            s = c[c.split == sp]
            dist = s.label.value_counts().reindex([1, 0, -1]).fillna(0).astype(int)
            hr = s.has_hedge.mean() * 100
            print(f"  {sp:5s} n={len(s):6d}  pos={dist[1]:6d} neg={dist[0]:6d} "
                  f"unc={dist[-1]:6d}  hedge={hr:4.1f}%")
        # hard-negative subset (label==0 & has_hedge) — the shortcut's failure set
        hn = c[(c.label == 0) & c.has_hedge]
        print(f"  hard-negatives (neg & hedged): {len(hn)} "
              f"({len(hn)/max(1,(c.label==0).sum())*100:.1f}% of negatives)")


if __name__ == "__main__":
    main()
