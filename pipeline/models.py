"""Models for the hedge-response CXR study.

  M1  image-only : DenseNet-121 -> linear (3-class)
  M2  text-only  : ClinicalBERT [CLS] -> linear (3-class)
  M3  fusion     : concat(image, text) -> MLP (3-class)

Runs where the CheXpert Plus images live; nothing here needs the images at
import time. Text encoder = emilyalsentzer/Bio_ClinicalBERT by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from transformers import AutoModel, AutoTokenizer

N_CLASSES = 3           # {neg, unc, pos}
TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class ImageEncoder(nn.Module):
    """DenseNet-121 backbone -> 1024-d feature (classifier removed).

    Global average pooling over the final feature map; the 1024-d result is
    the input to either the M1 head or the M3 fusion MLP."""
    def __init__(self, pretrained=True):
        super().__init__()
        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet121(weights=w)
        self.features = net.features
        self.out_dim = net.classifier.in_features   # 1024

    def forward(self, x):                            # x: (B,3,H,W)
        f = F.relu(self.features(x), inplace=True)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)   # (B,1024)


class TextEncoder(nn.Module):
    """ClinicalBERT -> 768-d [CLS] feature.

    Uses emilyalsentzer/Bio_ClinicalBERT, fine-tuned on MIMIC-III notes.
    The [CLS] token (position 0 of last_hidden_state) is the sentence-level
    representation fed to either the M2 head or the M3 fusion MLP."""
    def __init__(self, name=TEXT_MODEL):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        self.out_dim = self.bert.config.hidden_size   # 768

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]            # [CLS]


class Head(nn.Module):
    """Single linear layer mapping from feature dim to N_CLASSES logits.

    Used as the M1 and M2 classification head (no hidden layer: the encoder
    already produces a general-purpose feature).
    """
    def __init__(self, in_dim, n=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n)

    def forward(self, z):
        return self.fc(z)


class FusionMLP(nn.Module):
    """Two-layer MLP over concatenated image + text features.

    concat(1024-d image, 768-d text) -> Linear(512) -> ReLU -> Dropout ->
    Linear(3). Dropout(0.3) is mild regularisation; the encoders are
    pre-trained and carry most of the representational capacity.
    """
    def __init__(self, img_dim, txt_dim, hidden=512, n=N_CLASSES, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden), nn.ReLU(),
            nn.Dropout(p), nn.Linear(hidden, n))

    def forward(self, zi, zt):
        return self.net(torch.cat([zi, zt], dim=1))


class FusionModel(nn.Module):
    """M3: late concatenation of the image and text branches.

    forward() returns (logits, zi, zt) — the per-branch features are exposed
    so train.py can pull them out once per batch and reuse zi across text
    views at test time (avoids re-encoding the image three times).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.img = ImageEncoder(pretrained)
        self.txt = TextEncoder()
        self.head = FusionMLP(self.img.out_dim, self.txt.out_dim)

    def forward(self, image, input_ids, attn):
        zi = self.img(image)
        zt = self.txt(input_ids, attn)
        return self.head(zi, zt), zi, zt


def build_tokenizer(name=TEXT_MODEL):
    return AutoTokenizer.from_pretrained(name)
