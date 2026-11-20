"""Load the saved model and predict on new text. No retraining needed."""
import json
import os

import torch

from src.model import CharCNN, CharVocab
from src.preprocess import normalize


def load_model(model_dir="model"):
    """Load the trained model, vocabulary and settings (threshold, max length) from model_dir."""

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    vocab = CharVocab.load(os.path.join(model_dir, "vocab.json"))
    net = CharCNN(len(vocab), **cfg["model"])
    net.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    net.eval()
    return {"net": net, "vocab": vocab, "cfg": cfg}


@torch.no_grad()
def predict(bundle, text):
    """Return {'label': 0 or 1, 'probability': p} for one comment."""
    
    norm = normalize(text)
    x = torch.tensor([bundle["vocab"].encode(norm, bundle["cfg"]["max_len"])])
    p = torch.sigmoid(bundle["net"](x)).item()
    return {"label": int(p >= bundle["cfg"]["threshold"]), "probability": round(p, 4)}
