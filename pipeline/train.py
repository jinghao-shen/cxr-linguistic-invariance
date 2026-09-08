"""Train one model and dump test-set predictions. Designed for Kaggle/Colab (CUDA).

  python -m pipeline.train --cohort pipeline/cohorts/edema.parquet \
      --image-root /kaggle/input/.../CheXpert-v1.0-small --model m3 --out runs/edema_m3

--model in {m1 (image-only), m2 (text-only), m3 (fusion)}.
Saves <out>/preds_test.parquet with per-class probabilities under each text view
(h / c1 / p), consumed by placebo_contrast.py and justified_shift.py.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from pipeline.dataset import CXRDataset, load_split, class_weights, LABEL2IDX
from pipeline.models import ImageEncoder, TextEncoder, Head, FusionModel, build_tokenizer

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# fp16 autocast + loss scaling. DenseNet-121 and BERT-base are both small enough
# that fp32 leaves a T4/L4 idle; this is ~2x with no change to what is measured
# (every logit is cast back to fp32 before softmax, so all reported probabilities
# and AUROCs are fp32).
AMP = DEV == "cuda"


def amp_ctx():
    return torch.autocast("cuda", dtype=torch.float16, enabled=AMP)


# ----- model wrappers -------------------------------------------------------
class ImageOnly(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.img = ImageEncoder(pretrained)
        self.head = Head(self.img.out_dim)

    def forward(self, image, **_):
        return self.head(self.img(image))


class TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = TextEncoder()
        self.head = Head(self.txt.out_dim)

    def forward(self, ids, attn, **_):
        return self.head(self.txt(ids, attn))


def macro_auroc(y, proba):
    aucs = []
    for k in range(3):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            aucs.append(roc_auc_score(yk, proba[:, k]))
    return float(np.mean(aucs))


def pos_auroc(y, proba):
    """positive-vs-rest AUROC (pos class index = 2); the clinical head we
    early-stop on, so the near-random uncertain class can't stop training early."""
    yk = (y == 2).astype(int)
    if not 0 < yk.sum() < len(yk):
        return float("nan")
    return float(roc_auc_score(yk, proba[:, 2]))


def logits_for(model, kind, batch):
    img = batch["image"].to(DEV)
    if kind == "m1":
        return model(image=img)
    if kind == "m2":
        return model(ids=batch["ids_h"].to(DEV), attn=batch["m_h"].to(DEV))
    out, _, _ = model(img, batch["ids_h"].to(DEV), batch["m_h"].to(DEV))
    return out


def run_epoch(model, kind, loader, opt, cw, scaler=None):
    model.train()
    for b in loader:
        opt.zero_grad()
        y = b["label"].to(DEV)
        with amp_ctx():
            logits = logits_for(model, kind, b)
            loss = F.cross_entropy(logits, y, weight=cw)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(model, kind, loader, only_h=False):
    """Return dict of arrays: label, and proba under each text view available.

    `only_h` for the per-epoch validation pass: early stopping reads the h view
    alone, so scoring the masked/placebo views there would cost extra forward
    passes per epoch for numbers nothing looks at."""
    model.eval()
    # text views: h = original, c1 = hedge-masked (frame only, pathology name
    # kept), p = placebo (matched non-cue mask). M2 gets c1/p too, so the
    # text-only and fusion responses to the *same* edit are comparable without
    # an architecture gap.
    TEXT_VIEWS = {"h": ("ids_h", "m_h"), "c1": ("ids_c1", "m_c1"),
                  "p": ("ids_p", "m_p")}
    labels, views = [], {"h": []}
    if only_h:
        pass
    elif kind in ("m3", "m2"):
        views = {k: [] for k in TEXT_VIEWS}
    for b in loader:
        labels.append(b["label"].numpy())
        img = b["image"].to(DEV)
        with amp_ctx():
            if kind == "m1":
                out = {"h": model(image=img)}
            elif kind == "m2":
                out = {k: model(ids=b[i].to(DEV), attn=b[m].to(DEV))
                       for k, (i, m) in TEXT_VIEWS.items() if k in views}
            else:
                # The image encoding does not depend on the text view, so run
                # DenseNet once per batch and reuse it: calling the full forward
                # per view re-encoded every image three times at test-time.
                out = {}
                zi = model.img(img)
                for key, (ids, m) in TEXT_VIEWS.items():
                    if key not in views:
                        continue
                    zt = model.txt(b[ids].to(DEV), b[m].to(DEV))
                    out[key] = model.head(zi, zt)
        for key, logits in out.items():
            views[key].append(F.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(labels), {k: np.concatenate(v) for k, v in views.items()}


@torch.no_grad()
def image_features(model, loader):
    """Frozen image-encoder features + labels, for the linear probe of the
    fusion model's image branch."""
    model.eval()
    feats, labels = [], []
    for b in loader:
        with amp_ctx():
            z = model.img(b["image"].to(DEV))
        feats.append(z.float().cpu().numpy())
        labels.append(b["label"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def linear_probe(model, tr_loader, te_loader):
    """Fit a linear head on the model's frozen image features and return test
    pos-vs-rest AUROC. A conservative check of whether fusion under-trained the
    image branch: linear readout vs. a fully end-to-end M1."""
    Xtr, ytr = image_features(model, tr_loader)
    Xte, yte = image_features(model, te_loader)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    if 2 not in clf.classes_:
        return float("nan")
    p_pos = clf.predict_proba(Xte)[:, list(clf.classes_).index(2)]
    return float(roc_auc_score((yte == 2).astype(int), p_pos))


def build_model(kind):
    if kind == "m1":
        return ImageOnly()
    if kind == "m2":
        return TextOnly()
    return FusionModel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model", required=True, choices=["m1", "m2", "m3"])
    ap.add_argument("--out", required=True)
    # Training stochasticity only: head init, dropout, batch order. The
    # patient-level split lives in data.py and stays at its own fixed seed, so
    # every --seed sees the *same* test set and the runs remain comparable.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr-img", type=float, default=1e-4)
    ap.add_argument("--lr-bert", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="dataloader workers; the default caps at the CPU count "
                         "(Kaggle gives 4) because image decode is the bottleneck")
    ap.add_argument("--filter-existing", action="store_true",
                    help="drop rows whose image file is missing (subset runs)")
    args = ap.parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("device:", DEV, "| model:", args.model, "| amp:", AMP,
          "| workers:", args.num_workers, flush=True)

    tok = build_tokenizer()
    need_img = args.model != "m2"
    dfs = {sp: load_split(args.cohort, sp) for sp in ["train", "val", "test"]}
    cw = class_weights(dfs["train"]).to(DEV)

    def make(sp, train, persistent=False):
        ds = CXRDataset(dfs[sp], args.image_root, tok, max_len=args.max_len,
                        train=train, need_image=need_img,
                        filter_existing=args.filter_existing)
        nw = args.num_workers
        return DataLoader(ds, batch_size=args.bs, shuffle=train, num_workers=nw,
                          pin_memory=True,
                          # JPEG decode + resize on 2 workers starved the GPU; the
                          # loaders that get re-iterated every epoch also keep their
                          # workers alive so we stop paying respawn per epoch.
                          persistent_workers=persistent and nw > 0,
                          prefetch_factor=4 if nw > 0 else None)

    tr, va = make("train", True, persistent=True), make("val", False, persistent=True)
    te = make("test", False)
    model = build_model(args.model).to(DEV)

    # per-component learning rates
    params = []
    if hasattr(model, "img"):
        params.append({"params": model.img.parameters(), "lr": args.lr_img})
    if hasattr(model, "txt"):
        params.append({"params": model.txt.parameters(), "lr": args.lr_bert})
    params.append({"params": model.head.parameters(), "lr": args.lr_img})
    opt = torch.optim.AdamW(params)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best, best_state, patience, bad = -1, None, 5, 0
    t_start = time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        run_epoch(model, args.model, tr, opt, cw, scaler=scaler)
        t_tr = time.time() - t0
        yv, pv = predict(model, args.model, va, only_h=True)
        auc = pos_auroc(yv, pv["h"])
        # timings printed per epoch, flushed: a silent run is indistinguishable
        # from a hung one otherwise (see the pneumonia session that sat 75 min
        # with no output because the parent captured stdout block-buffered).
        print(f"epoch {ep}: val posAUROC {auc:.4f}  macro {macro_auroc(yv, pv['h']):.4f}"
              f"  [train {t_tr:.0f}s + val {time.time()-t0-t_tr:.0f}s]", flush=True)
        if auc > best:
            best, best_state, bad = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print("early stop", flush=True); break
    if best_state:
        model.load_state_dict(best_state)

    # dump test predictions
    print("scoring test set...", flush=True)
    yte, pte = predict(model, args.model, te)
    te_df = te.dataset.df.copy()
    rec = {"path_to_image": te_df["path_to_image"].values,
           "label": te_df["label"].values, "has_hedge": te_df["has_hedge"].values}
    for view, arr in pte.items():
        for k, name in enumerate(["neg", "unc", "pos"]):
            rec[f"p_{name}_{view}"] = arr[:, k]
    preds_df = pd.DataFrame(rec)
    preds_df.to_parquet(out / "preds_test.parquet")
    print("saved", out / "preds_test.parquet", flush=True)

    # Everything past this point is bookkeeping and diagnostics. An hour of
    # training already sits on disk above, so nothing down here is allowed to
    # raise: a NameError while assembling meta.json once cost three finished
    # models on a batch run that could not be restarted.
    try:
        meta = {"val_pos_auroc": best, "model": args.model, "n_test": len(yte),
                "cohort": args.cohort, "seed": args.seed, "epochs_run": ep + 1,
                "train_minutes": round((time.time() - t_start) / 60, 1)}
    except Exception as e:                       # noqa: BLE001 - never fatal here
        print(f"WARNING: could not assemble meta ({e!r})", flush=True)
        meta = {"model": args.model, "meta_error": repr(e)}
    # linear-probe of the image branch (models that have one)
    if hasattr(model, "img"):
        try:
            # a full extra forward pass over the train split: the slowest tail
            # step, so say so before disappearing into it.
            print("linear probe: extracting image features over train + test...", flush=True)
            meta["probe_pos_auroc"] = linear_probe(model, make("train", False), te)
            print(f"linear-probe image-branch pos-AUROC: {meta['probe_pos_auroc']:.4f}", flush=True)
        except Exception as e:                   # noqa: BLE001
            print(f"WARNING: linear probe failed ({e!r})", flush=True)
            meta["probe_error"] = repr(e)
    try:
        json.dump(meta, open(out / "meta.json", "w"), indent=2)
    except Exception as e:                       # noqa: BLE001
        print(f"WARNING: could not write meta.json ({e!r})", flush=True)
    print("done:", args.model, flush=True)


if __name__ == "__main__":
    main()
