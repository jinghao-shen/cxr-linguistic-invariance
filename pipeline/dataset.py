"""Torch dataset for the hedge-response fusion experiments.

Each item carries the image plus three tokenised text views so a single
trained model can be scored under all of them at eval time without reloading:

  h        original indication (identity)
  c1(h)    hedge-masked: epistemic frame replaced with [MASK], pathology kept
  π(h)     placebo: matched neutral spans masked, no cue or pathology touched

Labels {0 neg, -1 unc, 1 pos} are remapped to indices {neg:0, unc:1, pos:2}
so cross-entropy sees a contiguous [0, N_CLASSES) target.
"""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

LABEL2IDX = {0: 0, -1: 1, 1: 2}          # neg, unc, pos
IDX2NAME = ["neg", "unc", "pos"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size=224, train=False):
    """ImageNet-normalised transforms. Train augments with crop + rotation;
    eval uses a deterministic center crop.  Grayscale CXR -> 3-channel copy
    so the ImageNet-pretrained DenseNet-121 backbone sees the expected shape.
    """
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(int(image_size * 1.14)),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.RandomRotation(10),
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _tok(tokenizer, text, max_len):
    """Tokenise `text` with padding/truncation; return (input_ids, attn_mask)
    as 1-D LongTensors ready to batch."""
    e = tokenizer(text or "", truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
    return e["input_ids"][0], e["attention_mask"][0]


class CXRDataset(Dataset):
    """CheXpert Plus study dataset: image + three text views per item.

    Args:
        df:              split-specific DataFrame (from load_split)
        image_root:      path prefix prepended to df["path_to_image"]
        tokenizer:       HuggingFace tokenizer (Bio_ClinicalBERT)
        max_len:         BERT sequence length (default 48; 99th pct < 40 tokens)
        train:           True -> data augmentation; False -> deterministic crop
        need_image:      False for M2 (text-only), skips JPEG decode
        filter_existing: drop rows whose image file is missing (subset runs)
    """
    def __init__(self, df, image_root, tokenizer,
                 max_len=48, image_size=224, train=False, need_image=True,
                 filter_existing=False):
        self.image_root = Path(image_root)
        self.tok = tokenizer
        self.max_len = max_len
        self.need_image = need_image
        self.tf = build_transforms(image_size, train=train)
        df = df.reset_index(drop=True)
        if filter_existing:
            keep = df["path_to_image"].map(lambda p: (self.image_root / p).exists())
            df = df[keep].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # ---- image ----------------------------------------------------------
        if self.need_image:
            img = Image.open(self.image_root / r["path_to_image"]).convert("L")
            image = self.tf(img)
        else:
            image = torch.zeros(3, 1, 1)      # placeholder for text-only models
        # ---- three text views -----------------------------------------------
        ids_h, m_h = _tok(self.tok, r["h"], self.max_len)
        ids_c1, m_c1 = _tok(self.tok, r["h_masked"], self.max_len)
        # placebo column is present in _ext cohorts; fall back to h (identity)
        # for cohorts built without mask_placebo (no-op for cue-free rows anyway).
        pl = r["h_placebo"] if "h_placebo" in r else r["h"]
        ids_p, m_p = _tok(self.tok, pl, self.max_len)
        return {
            "image": image,
            "ids_h": ids_h, "m_h": m_h,
            "ids_c1": ids_c1, "m_c1": m_c1,
            "ids_p": ids_p, "m_p": m_p,
            "label": torch.tensor(LABEL2IDX[int(r["label"])], dtype=torch.long),
            "idx": i,
        }


def load_split(cohort_path, split):
    df = pd.read_parquet(cohort_path)
    return df[df.split == split].reset_index(drop=True)


def class_weights(df):
    """Inverse-frequency weights for the 3-class CE loss (handles class imbalance)."""
    n = df["label"].map(LABEL2IDX).value_counts().reindex([0, 1, 2]).fillna(0)
    w = len(df) / (3.0 * n.clip(lower=1))
    return torch.tensor(w.values, dtype=torch.float32)
