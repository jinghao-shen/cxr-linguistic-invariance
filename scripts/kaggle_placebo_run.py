# ===== Kaggle cell: placebo-controlled hedge audit, one pathology per run =====
#
# Change PATHOLOGY below and rerun. Everything else is parameterised.
#
# ---------------------------------------------------------------------------
# WHAT THIS TESTS
#   c1 masks the epistemic frame ("rule out edema" -> "[MASK] edema"). The model
#   shifts. That is only a shortcut if the shift is specific to hedges, so the
#   placebo view masks a matched number/length of NEUTRAL words -- never a cue,
#   never a pathology mention. gap = delta(c1) - delta(placebo) is the
#   hedge-specific part.
#
#   Cells:  A = the hedge is about the target pathology   (legitimate signal)
#           B = the hedge is about a DIFFERENT disease    (carries no information
#               about the target, so ANY response is unjustifiable -- the test)
#           C = hedged, no disease named
#
# RESULT BEING REPLICATED -- pleural effusion, pooled over cells, n=13,610
#   justified shift is ~0 on every class; the model's is not:
#     class   justified                model                   epsilon
#     neg     -0.0012 [-.0166,+.0127]  -0.0181 [-.0224,-.0147]  -0.0169 [-.0310,-.0013]
#     unc     -0.0032 [-.0174,+.0084]  -0.0239 [-.0284,-.0186]  -0.0208 [-.0342,-.0066]
#     pos     +0.0053 [-.0126,+.0223]  +0.0420 [+.0385,+.0451]  +0.0367 [+.0203,+.0547]
#   i.e. masking the hedge moves ~3.7pp of mass out of {neg, unc} into pos, none of
#   it warranted once the image is conditioned on. Placebo net -0.2% (13 up/15 down)
#   against hedge net +2.4% [+1.3%,+3.5%]. Edema agrees in direction, CIs cross 0.
#
# PRE-SPECIFIED PREDICTIONS FOR CONSOLIDATION -- written before the run, not to be
# edited after seeing output. Consolidation differs from effusion in two ways that
# matter: its uncertain class is 39.7% of the cohort against effusion's 5.4%, and
# its positive base rate is 0.347 against 0.740.
#   C1  placebo net ~ 0 and directionally symmetric     -> 95% CI includes 0
#       SANITY CHECK. If this fails the instrument is invalid on this cohort and
#       C2-C5 are uninterpretable whatever they say.
#   C2  cell-B gap != 0 on the pos class                -> 95% CI excludes 0
#       (hedge-specificity replicates)
#   C3  sign pattern repeats: eps_pos > 0, eps_unc < 0  -> mass still moves out of
#       uncertain into positive
#   C4  |eps_unc| LARGER than effusion's 0.0208         -> the directional test of
#       our reading. We claim hedging holds the model at "uncertain"; consolidation
#       has 7x more uncertain mass to hold, so the effect should grow. If |eps_unc|
#       comes back at or below effusion's, the "holds it at uncertain" account is
#       wrong even if C2/C3 pass, and the finding is a generic push toward positive.
#   C5  eps_pos CI excludes 0                           -> full replication
#       Cell B here is 425 studies against effusion's 1072, so C5 may fail on power
#       alone; C3 and C4 are the predictions that carry interpretation.
#
# WHY THIS PATHOLOGY (selected on sample size and label base rates ONLY --
# never on any model output, so the choice is a design decision, not a result):
#   Pleural Effusion  cell B 1052 (1.8x edema), text-only pos-AUROC .788 vs
#                     edema's .678, anchor gap .036, base P(pos) .740, n=90k.
#                     Cost: uncertain is only 5.4% of labels.
#   Consolidation     best 3-class balance 20/40/40; text predicts the UNCERTAIN
#                     class at .714. Cost: cell B is 390.
#   Pneumonia         NOT USABLE: cell B = 55, and its 7 pos/neg rows sit at
#                     P(pos)=.857 vs base .641, so the anchor fails. P1/P2/P4
#                     cannot be tested there; only P3/P5 can.
#
# Settings: Accelerator = GPU T4 x2, Internet = ON.
# REQUIRES the 2026-07-31 train.py (t_start fix, hardened tail, --seed).
# ---------------------------------------------------------------------------

PATHOLOGY = "consolidation"         # pleural_effusion | consolidation | edema | pneumonia
SEEDS = [0]                         # e.g. [0, 1, 2] to repeat; each seed = a full re-train
MODELS = ["m3", "m2", "m1"]         # m3 first: if the session dies, keep what matters

import os, glob, shutil

CODE_DIR = "/kaggle/input/datasets/jinghaoshen/pipline1"
COHORT_SRC = f"/kaggle/input/datasets/jinghaoshen/cohorts/{PATHOLOGY}_ext.parquet"

# target-pathology synonyms; everything else in DISEASE_RE counts as "another disease"
TARGET_RE = {
    "edema": r"\b(?:edema|chf|congestive heart failure|fluid overload|volume overload)",
    "pleural_effusion": r"\b(?:effusion|pleural fluid|hemothorax)",
    "consolidation": r"\b(?:consolidat|infiltrat)",
    "pneumonia": r"\b(?:pneumonia|pna|infiltrat|consolidat|infection|septic)",
}[PATHOLOGY]

# ---------- fail fast: right code, right cohort, right GPU --------------------
assert os.path.exists(f"{CODE_DIR}/train.py"), f"no train.py in {CODE_DIR}"
assert os.path.exists(COHORT_SRC), f"missing {COHORT_SRC} -- upload the rebuilt cohort"

src = open(f"{CODE_DIR}/train.py").read()
assert '"p": ("ids_p", "m_p")' in src, "OLD train.py: no placebo view. Re-upload it."
assert '"c1": ("ids_c1", "m_c1")' in src, "OLD train.py: no explicit c1/c2 views"
assert "t_start = time.time()" in src, "OLD train.py: the t_start crash is still there"
assert '"--seed"' in src or "'--seed'" in src, "OLD train.py: no --seed support"
assert "ids_p" in open(f"{CODE_DIR}/dataset.py").read(), "OLD dataset.py"
# data.py is imported by the contrast at the bottom (DISEASE_RE), so it must be
# uploaded too -- it used to be optional, since training reads the baked cohort.
assert os.path.exists(f"{CODE_DIR}/data.py"), (
    f"no data.py in {CODE_DIR} -- the contrast needs DISEASE_RE from it")
assert "DISEASE_RE" in open(f"{CODE_DIR}/data.py").read(), "OLD data.py: no DISEASE_RE"

import pandas as pd
_coh = pd.read_parquet(COHORT_SRC)
for col in ("h_placebo", "placebo_ok"):
    assert col in _coh.columns, f"cohort has no `{col}` -- it is the old build"
_te = _coh[_coh.split == "test"]
print(f"{PATHOLOGY}: n={len(_coh)}  test={len(_te)}  "
      f"hedged={int(_te.has_hedge.sum())}  placebo_ok={int(_te.placebo_ok.sum())}")

import torch
assert torch.cuda.is_available(), "no GPU: Settings -> Accelerator = GPU T4 x2"
_cc = torch.cuda.get_device_capability(0)
assert _cc >= (7, 0), (f"{torch.cuda.get_device_name(0)} is sm_{_cc[0]}{_cc[1]}; this "
                       f"torch supports {torch.cuda.get_arch_list()}. Use GPU T4 x2.")
print("gpu:", torch.cuda.get_device_name(0))


# Bounded-depth scan: glob("/kaggle/input/**", recursive=True) descends into every
# one of the ~200k patient directories and takes many minutes.
def _is_root(p):
    tr = os.path.join(p, "train")
    if not os.path.isdir(tr):
        return False
    with os.scandir(tr) as it:
        return any(e.name.startswith("patient") for e in it)


def find_image_root(base="/kaggle/input", max_depth=4):
    frontier = [(base, 0)]
    while frontier:
        path, depth = frontier.pop(0)
        if _is_root(path):
            return path
        if depth >= max_depth:
            continue
        try:
            with os.scandir(path) as it:
                for e in it:
                    if e.is_dir() and not e.name.startswith("patient"):
                        frontier.append((e.path, depth + 1))
        except PermissionError:
            pass
    return None


IMAGE_ROOT = find_image_root()
assert IMAGE_ROOT, f"no CheXpert image tree under /kaggle/input {os.listdir('/kaggle/input')}"
print("image root:", IMAGE_ROOT)

# ---------- assemble a real `pipeline` package in the writable dir ------------
dst = "/kaggle/working/pipeline"
shutil.rmtree(dst, ignore_errors=True)
os.makedirs(f"{dst}/cohorts", exist_ok=True)
for f in glob.glob(f"{CODE_DIR}/*.py"):
    shutil.copy(f, dst)
open(f"{dst}/__init__.py", "a").close()
shutil.copy(COHORT_SRC, f"{dst}/cohorts/{PATHOLOGY}_ext.parquet")
os.chdir("/kaggle/working")
COHORT = f"pipeline/cohorts/{PATHOLOGY}_ext.parquet"
print("modules:", sorted(os.listdir(dst)))

status = {}


def run(model, seed):
    # Never raise: under Save & Run All an exception discards every later cell, so
    # one model's crash would also cost the others.
    tag = f"{PATHOLOGY}_{model}" + (f"_s{seed}" if seed else "")
    cmd = (f"python -u -m pipeline.train --cohort {COHORT} --image-root {IMAGE_ROOT} "
           f"--model {model} --seed {seed} --out runs/{tag}")
    print("\n>>>", cmd, flush=True)
    rc = os.system(cmd)
    status[tag] = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"[{tag}] {status[tag]}", flush=True)


for _seed in SEEDS:
    for _m in MODELS:
        run(_m, _seed)

print("\n=== run status ===")
for k, v in status.items():
    print(f"  {k}: {v}")
print("\nfiles produced:")
for dp, _, fs in os.walk("runs"):
    for f in fs:
        pth = os.path.join(dp, f)
        print(f"  {pth}  ({os.path.getsize(pth)/1e6:.1f} MB)")

# ---------- pre-specified contrast, inline so the answer is in the notebook ---
import numpy as np

m3_dir = f"runs/{PATHOLOGY}_m3"
if not os.path.exists(f"{m3_dir}/preds_test.parquet"):
    print("\nno m3 predictions -- nothing to contrast. Download runs/ and rerun m3.")
else:
    from pipeline.data import DISEASE_RE
    d = pd.read_parquet(f"{m3_dir}/preds_test.parquet").merge(
        _coh[["path_to_image", "h", "placebo_ok", "deid_patient_id"]], on="path_to_image")
    t = d["h"].astype(str)
    is_a = t.str.contains(TARGET_RE, case=False, regex=True)
    is_b = t.str.contains(DISEASE_RE) & ~is_a
    d["cell"] = np.where(is_a, "A: target", np.where(is_b, "B: other dx", "C: no dx"))
    d["d_c1"] = d["p_pos_c1"] - d["p_pos_h"]
    d["d_pl"] = d["p_pos_p"] - d["p_pos_h"]
    d["gap"] = d["d_c1"] - d["d_pl"]

    def boot(s, col, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        g = list(s.groupby("deid_patient_id", sort=False).indices.values())
        v = s[col].values
        a = np.array([v[np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])].mean()
                      for _ in range(n)])
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    e = d[d.placebo_ok.astype(bool)]
    thr = float(np.quantile(d.p_pos_h, 1 - (d.label == 1).mean()))
    print(f"\n=== {PATHOLOGY}: placebo contrast (placebo_ok n={len(e)}, thr={thr:.3f}) ===")
    print(f"  {'cell':12s} {'n':>5s} {'d(c1)':>9s} {'d(plac)':>9s} {'gap':>9s} "
          f"{'95% CI of gap':>20s} {'net(hedge)':>11s} {'95% CI':>18s}")
    gaps = {}
    for c in sorted(e.cell.unique()):
        s = e[e.cell == c]
        if len(s) < 20:
            print(f"  {c:12s} {len(s):5d}  (too few)")
            continue
        lo, hi = boot(s, "gap")
        was, now = s.p_pos_h > thr, s.p_pos_c1 > thr
        ss = s.assign(net=(now & ~was).astype(float) - (was & ~now).astype(float))
        nlo, nhi = boot(ss, "net")
        gaps[c] = s.gap.mean()
        print(f"  {c:12s} {len(s):5d} {s.d_c1.mean():+9.4f} {s.d_pl.mean():+9.4f} "
              f"{s.gap.mean():+9.4f} [{lo:+.4f},{hi:+.4f}] {ss.net.mean()*100:+10.1f}% "
              f"[{nlo*100:+.1f}%,{nhi*100:+.1f}%]")
    if "A: target" in gaps and "B: other dx" in gaps and gaps["A: target"] > 1e-9:
        print(f"\n  spurious fraction gap_B/gap_A = {gaps['B: other dx']/gaps['A: target']:.2f}"
              "   (edema: 0.52 [0.35, 0.74])")
    print("\n  P1 cell-B gap CI excludes 0?   P4 cell-B net CI excludes 0?")
    print("  P5 sanity: placebo net must be ~0 -- rerun placebo_contrast.py locally for it.")
