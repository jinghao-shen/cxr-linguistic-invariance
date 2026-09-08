# Quantifying Unwarranted Response to Hedging Language in Multimodal Chest X-ray Models

*A placebo- and image-conditional audit across three chest-radiograph pathologies*

Jinghao Shen · NC State GEARS Research · Mentor: Dr. Min Chi

---

## Research Question

Chest X-ray fusion models read the clinical indication alongside the radiograph. Indications routinely hedge a suspected diagnosis — *"rule out pneumothorax," "concern for infiltrate"* — encoding the clinician's pre-test uncertainty rather than a radiographic finding. The question this project asks is:

> When the model reads a hedge, is its response scoped to the disease that hedge concerns — or does it react to the epistemic register itself, regardless of what the clause is about?

The answer matters because phrasing conventions differ across hospitals, services, and individual clinicians in ways that carry no information about the patient.

---

## Key Finding

Masking the epistemic frame in a clause about a **different** disease still raises the model's predicted probability of the target pathology by **+0.025 to +0.045**, replicating across three pathologies (pleural effusion, consolidation, pulmonary edema). All three patient-clustered bootstrap 95% confidence intervals exclude zero; the pooled estimate is **+0.0297** [+0.0267, +0.0327] (*n* = 3,218). A placebo edit masking matched neutral words moves predictions by at most 0.011, and the effect survives on indications naming no disease at all — confirming the response tracks the epistemic register, not any disease word. It reaches decisions: 2–4% of studies cross a prevalence-matched threshold in two of three cohorts, almost entirely in one direction.

Standard evaluation metrics cannot see this. The AUROC-based version of the same contrast returns +0.0003 where the signed probability shift returns +0.030, because AUROC is rank-based and blind to a shift that moves every prediction the same way.

---

## Models and Data

**Dataset.** CheXpert Plus (223,462 image–report pairs). Text modality is the pre-imaging clinical indication, not the diagnostic report — this removes the confound that the dataset labels (CheXbert) are a deterministic function of the report. The indication is written under two section headers; recovering both enlarges the pleural effusion cohort from 69,505 to 90,016 studies. Splits are patient-disjoint (70/15/15 by `deid_patient_id`).

**Pathologies.** Three: pleural effusion (primary, *n* = 90,016), pulmonary edema (replication, *n* = 58,561), and consolidation (pre-registered replication, *n* = 46,317 — predictions fixed in writing before the model ran).

**Model family.**

| Model | Architecture | Description |
|-------|-------------|-------------|
| M1 | DenseNet-121 → linear | Image-only baseline |
| M2 | ClinicalBERT [CLS] → linear | Text-only ceiling |
| M3 | concat(image, text) → MLP-512 | Fusion (late concatenation) |

DenseNet-121 is ImageNet-pretrained; its classifier is removed and global average pooling gives a 1024-d image feature. ClinicalBERT ([`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)) gives a 768-d [CLS] feature. Training uses AdamW (lr 2×10⁻⁵ for BERT, 1×10⁻⁴ for DenseNet and heads), class-weighted cross-entropy, batch size 32, up to 15 epochs, early stopping on validation pos-AUROC (patience 5). Platform: Kaggle GPU T4 ×2.

---

## Experimental Design

### Text Views

Every trained model is scored under three text views of the same indication — no retraining:

| View | Operator | Description |
|------|----------|-------------|
| `h` | identity | original indication as written |
| `c1(h)` | frame mask | replace epistemic frame with `[MASK]`, keep pathology name |
| `π(h)` | placebo | mask same number/length of neutral word spans |

The placebo is the decisive control: it matches `c1` on every surface property (count of `[MASK]` tokens, span lengths, identity on cue-free text) and differs only in *what* is masked, so the contrast `gap = delta(c1) - delta(placebo)` isolates the hedge-specific part from generic perturbation sensitivity.

### Cell Taxonomy

Hedged studies are partitioned by what the hedged clause is *about*:

- **Cell A** — hedge concerns the **target pathology** (legitimate signal)
- **Cell B** — hedge concerns a **different disease** (the decisive test; no information about target)
- **Cell C** — hedge present, **no disease named** (second leg)

Cell B is the primary test: the target pathology is never mentioned, so a nonzero gap cannot be attributed to disease-specific content.

### Estimands

The estimands are signed probability shifts, not AUROC differences:

1. **gap_k** (Eq. 1): `E[p_k(x, c1(h)) − p_k(x, h)] − E[p_k(x, π(h)) − p_k(x, h)]` — hedge-specific shift per class
2. **Δ_just** (Eq. 2): G-computation ATT over a spline in M1's log-odds — justified shift after conditioning on image
3. **Decision net**: rate studies cross a prevalence-matched threshold, reported directionally (hedge mask vs. placebo)

All intervals are patient-clustered bootstrap percentile CIs (2,000 resamples).

---

## Repository Structure

```
cxr-linguistic-invariance/
├── pipeline/                      # installable Python package
│   ├── data.py                    # hedge lexicon, masking operators, cohort builder
│   ├── dataset.py                 # CXRDataset: image + three text views (h/c1/π)
│   ├── models.py                  # ImageEncoder, TextEncoder, FusionModel (M1/M2/M3)
│   ├── train.py                   # training loop; dumps preds_test.parquet
│   ├── dose_response.py           # cue-count bins -> mean|Δp| (limitations check)
│   ├── cohorts/                   # [gitignored] patient-split parquets
│   └── analysis/
│       ├── placebo_contrast.py    # gap = delta(c1) − delta(placebo), per cell
│       └── justified_shift.py     # G-computation ATT, patient-clustered bootstrap
├── figures/
│   ├── make_architecture.py       # two-branch architecture diagram
│   ├── make_paper_charts.py       # forest plot, flips chart, epsilon panel (paper style)
│   └── make_report_figs.py        # label distribution, hedge prevalence charts
├── scripts/
│   ├── kaggle_placebo_run.py      # Kaggle notebook: trains M1/M2/M3, inline contrast
│   ├── compute_table.py           # cell A/B/C counts and examples
│   ├── grab_example_images.py     # fetch two example images from Kaggle
│   └── pick_examples.py           # select real studies for the example figure
├── requirements.txt
└── README.md
```

---

## Hedge Lexicon

The lexicon (`pipeline/data.py: HEDGE_CUES`) contains 28 epistemic-frame cues drawn from the CheXpert/NegBio uncertainty rules, Panicek & Hricak (2016), and a frequency scan of the 138,857 non-empty indications. It is frozen before any model is trained. Under this lexicon, 12.8% of all indications carry at least one hedge cue; the rate is higher among negative labels (15–17% for the studied pathologies, 34.5% for pneumonia), reflecting that clinicians hedge most when ruling a diagnosis out.

---

## Results Summary

| Cohort | Cell B *n* | shift under c1 | shift under π | gap [95% CI] |
|--------|-----------|----------------|---------------|--------------|
| Pleural effusion | 1,072 | +0.0368 | +0.0031 | **+0.0337** [+0.0286, +0.0386] |
| Consolidation (pre-reg.) | 425 | +0.0377 | −0.0075 | **+0.0452** [+0.0373, +0.0535] |
| Pulmonary edema | 589 | +0.0144 | −0.0110 | **+0.0255** [+0.0181, +0.0325] |

Pooled (3,218 studies): **+0.0297** [+0.0267, +0.0327].

Decision threshold crossings in cell B (prevalence-matched): effusion +2.4% [+1.3%, +3.5%], consolidation +3.7% [+1.9%, +5.7%]. Placebo nets ≈ 0 in both.

The raw hedge–label association in cell B (+0.125 for effusion) collapses to +0.001 [−0.014, +0.017] once the radiograph is conditioned on (G-computation), leaving no signal for the model's +0.037 shift to be tracking.

---

## Running the Pipeline

**Step 1 — build cohort parquets** (runs locally, no images needed):
```bash
python -m pipeline.data --pathologies "Pleural Effusion" Edema Consolidation \
    --suffix _ext
```

**Step 2 — train on Kaggle** (GPU T4 ×2, Internet ON):

Upload the `pipeline/` directory as a Kaggle dataset, then run `scripts/kaggle_placebo_run.py` as a notebook cell with `PATHOLOGY` set to the desired pathology. This trains M1, M2, M3 in sequence and outputs `runs/<tag>/preds_test.parquet`.

**Step 3 — post-hoc analysis** (download `runs/` and run locally):
```bash
# primary contrast
python -m pipeline.analysis.placebo_contrast \
    --m3 runs/effusion_m3 --cohort pipeline/cohorts/pleural_effusion_ext.parquet

# G-computation justified shift
python -m pipeline.analysis.justified_shift \
    --m1 runs/effusion_m1 --m3 runs/effusion_m3 \
    --cohort pipeline/cohorts/pleural_effusion_ext.parquet

# dose-response check (limitations)
python -m pipeline.dose_response \
    --m3 runs/effusion_m3 --cohort pipeline/cohorts/pleural_effusion_ext.parquet
```

**Step 4 — figures**:
```bash
python figures/make_report_figs.py --out figs/
python figures/make_paper_charts.py --out figs/
```

---

## References

- CheXpert Plus: Chambon et al., arXiv:2405.19538, 2024.
- CheXbert: Irvin et al., AAAI 2019 (label source).
- ClinicalBERT: Alsentzer et al., Clinical NLP Workshop (NAACL), 2019.
- DenseNet: Huang et al., CVPR 2017.
- Shortcut learning: Geirhos et al., Nature Machine Intelligence 2020.
